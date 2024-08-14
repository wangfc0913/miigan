import copy

import torch.nn as nn
import torch


class ChannelAttentionModule(nn.Module):
    def __init__(self, inc, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = inc // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=inc, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=inc)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        # print(f'ChannelAttentionModule x.shape: {x.shape}')
        # print(f'ChannelAttentionModule x.shape: {self.avg_pool(x).view(x.size(0), -1).shape}')
        avg_out = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        max_out = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        out = self.sigmoid(avg_out + max_out)
        out = out.expand_as(x)
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self, inc):
        super(SpatialAttentionModule, self).__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, inc, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.shared_conv2d(att)
        return att


class HSCAMBlock(nn.Module):
    def __init__(self, hidden_size):
        super(HSCAMBlock, self).__init__()
        self.channel_attention = ChannelAttentionModule(hidden_size)
        self.spatial_attention = SpatialAttentionModule(hidden_size)
        self.conv = nn.Conv2d(hidden_size * 2, hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # print(f'HSCAMBlock x.shape: {x.shape}')
        residual = x
        outc = self.channel_attention(x)
        outs = self.spatial_attention(x)
        assert outc.shape == outs.shape, f'{outc, outs} tensors must have the same shape'
        out = torch.cat([outc, outs], dim=1)
        out = self.sigmoid(self.conv(out))
        out = out * outc
        out = out * outs
        out = out + residual
        return out.permute(0, 2, 3, 1)


class HSCAMLayer(nn.Module):
    def __init__(self, config):
        super(HSCAMLayer, self).__init__()
        self.layers = nn.ModuleList()
        for idx_layer in range(len(config.mamba.encoder_depths)):
            layer = HSCAMBlock(config.mamba.embed_dims[idx_layer])
            self.layers.append(copy.deepcopy(layer))

    def forward(self, feature_list):

        assert len(feature_list) == len(self.layers), "feature_list must not be None"

        result = []
        for idx, layer in enumerate(self.layers):
            # print(f'current layer: {idx}')
            # print(f'feature_list[{idx}]: {feature_list[idx].shape}')
            t = layer(feature_list[idx])
            result.append(t)

        return result


class FeatureDiffAndProd(nn.Module):
    def __init__(self, config):
        super(FeatureDiffAndProd, self).__init__()

        self.config = config

        self.use_conv = True

        if config.use_hscam:
            self.tf_layers = nn.ModuleList()
            for idx_layer in range(len(config.mamba.encoder_depths)):
                # print(idx_layer)
                # print(config.mamba.embed_dims[idx_layer])
                layer = HSCAMBlock(config.mamba.embed_dims[idx_layer])
                self.tf_layers.append(copy.deepcopy(layer))

            self.vm_layers = nn.ModuleList()
            for idx_layer in range(len(config.mamba.encoder_depths)):
                layer = HSCAMBlock(config.mamba.embed_dims[idx_layer])
                self.vm_layers.append(copy.deepcopy(layer))

        if self.use_conv:
            self.conv_layers = nn.ModuleList()
            for idx_layer in range(len(config.mamba.encoder_depths)):
                layer = nn.Conv2d(config.mamba.embed_dims[idx_layer], config.mamba.embed_dims[idx_layer],
                                  kernel_size=3, padding=1)
                self.conv_layers.append(copy.deepcopy(layer))

        self.skip_conv_layers = nn.ModuleList()
        for idx_layer in range(len(config.mamba.encoder_depths)):
            layer = nn.Conv2d(config.mamba.embed_dims[idx_layer] * 2, config.mamba.embed_dims[idx_layer],
                              kernel_size=3, padding=1)
            self.skip_conv_layers.append(copy.deepcopy(layer))

    def forward(self, tf_feature_list, vm_feature_list):
        assert len(tf_feature_list) == len(vm_feature_list), \
            "The dimensions of the two input parameters must be consistent."

        if self.config.use_hscam:
            df = []
            for idx, layer in enumerate(self.tf_layers):
                t = layer(tf_feature_list[idx])
                df.append(t)

            pf = []
            for idx, layer in enumerate(self.vm_layers):
                t = layer(vm_feature_list[idx])
                pf.append(t)

            tf_feature_list = df
            vm_feature_list = pf

        diffs = []
        prods = []
        for idx, (t, v) in enumerate(zip(tf_feature_list, vm_feature_list)):
            diff = torch.abs(v - t)
            prod = torch.mul(v, t)
            if self.use_conv:
                conv = self.conv_layers[idx]
                diff = diff.permute(0, 3, 1, 2).contiguous()
                prod = prod.permute(0, 3, 1, 2).contiguous()
                diff = conv(diff)
                prod = conv(prod)
                diff = diff.permute(0, 2, 3, 1).contiguous()
                prod = prod.permute(0, 2, 3, 1).contiguous()
            diffs.append(diff)
            prods.append(prod)

        dp_convs = []
        for idx, (d, p) in enumerate(zip(diffs, prods)):
            c = torch.cat((d, p), 3)
            c = c.permute(0, 3, 1, 2).contiguous()
            skip_conv = self.skip_conv_layers[idx]
            c = skip_conv(c)
            c = c.permute(0, 2, 3, 1).contiguous()
            dp_convs.append(c)

        return dp_convs, diffs, prods

# Test...
# batch_size, in_channels, H, W = 4, 256, 512, 512
# x = torch.randn(batch_size, in_channels, H, W)
# model = HSCAM(layer_chans=[64, 128, 256, 512], layer=2)
# # print(model)
# total = sum([param.nelement() for param in model.parameters()])
# # 精确地计算：1MB=1024KB=1048576字节
# print('Number of parameter: % .4fM' % (total / 1e6))
# out = model(x)
# print("Output shape:", out.shape)

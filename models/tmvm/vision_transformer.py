# coding=utf-8
import copy
import math

from torch.nn import Dropout, Softmax, Linear, LayerNorm
from .utils import *

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, hidden_size):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class TFBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super(TFBlock, self).__init__()

        self.hidden_size = config.mamba.embed_dims[layer_idx]
        self.next_block_dim = config.mamba.embed_dims[layer_idx]
        if 0 < layer_idx < len(config.mamba.embed_dims) - 1:
            self.next_block_dim = config.mamba.embed_dims[layer_idx + 1]
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(config, self.hidden_size)
        self.attn = Attention(config, self.hidden_size)
        self.block_tail_linear = Linear(self.hidden_size, self.next_block_dim)
        self.downSample = PatchMerging2D if (layer_idx < len(config.mamba.embed_dims) - 1) else None

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        if self.downSample is not None:
            x = rearrange(x, 'batch (h w) c -> batch h w c', h=int(x.size(1) ** 0.5))
            x = self.downSample(dim=self.hidden_size)(x)
            x = rearrange(x, 'batch h w c -> batch (h w) c')

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.mamba.embed_dims[-1], eps=1e-6)
        for idx_layer in range(len(config.mamba.encoder_depths)):
            layer = TFBlock(config, idx_layer)
            self.layers.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        feature_list = []
        for layer_block in self.layers:
            feature_list.append(hidden_states)
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, feature_list


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)

    def forward(self, x):
        x = dim_conversion(x)
        encoded, feature_list = self.encoder(x)  # (B, n_patch, hidden)
        return encoded, feature_list


# class VisionTransformer(nn.Module):
#     def __init__(self, config):
#         super(VisionTransformer, self).__init__()
#         self.transformer = Transformer(config)
#
#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         encoded, feature_list = self.transformer(x)  # (B, n_patch, hidden)
#         return encoded, feature_list
#
#
# CONFIGS = {
#     'training': configs.get_train_config(),
#     'testing': configs.get_test_config(),
# }

# cf = CONFIGS['training']
# batch_size, in_channels, H, W = 4, 3, 512, 512
# x = torch.randn(batch_size, in_channels, H, W)
#
# model = VisionTransformer(cf)
# total = sum([param.nelement() for param in model.parameters()])
# # 精确地计算：1MB=1024KB=1048576字节
# print('VisionTransformer Number of parameter: % .4fM' % (total / 1e6))
# encoded, feature_list = model(x)
# print("VisionTransformer Output shape:", encoded.shape)
# for f in feature_list:
#     print("VisionTransformer shape:", f.shape)

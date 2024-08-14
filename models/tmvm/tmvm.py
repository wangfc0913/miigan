import torch
from torch import nn

from .vision_mamba import VMEncoder, VMDecoder
from .vision_transformer import Transformer

from .utils import GenPatchEmbed2D, DisPatchEmbed2D, batch_rearrange, BottleneckModule, _rearrange
from .hscam import HSCAMLayer, FeatureDiffAndProd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TMVMGenerator(nn.Module):
    def __init__(self, config):
        super(TMVMGenerator, self).__init__()
        self.config = config
        self.patch_embed = GenPatchEmbed2D(config)
        self.vm_encoder = VMEncoder(config)
        self.vm_decoder = VMDecoder(config)

        if self.config.use_hscam:
            self.hscam_layer = HSCAMLayer(config)

    def forward(self, x):
        x = self.patch_embed(x)
        x, skip_list = self.vm_encoder(x)
        # print(x.size())
        # print(skip_list[-1].size())
        if self.config.use_hscam:
            skip_list = self.hscam_layer(skip_list)
        x = self.vm_decoder(x, skip_list)
        return x


class TMVMDiscriminator(nn.Module):
    def __init__(self, config):
        super(TMVMDiscriminator, self).__init__()
        self.config = config
        self.patch_embed = DisPatchEmbed2D(config)
        self.tf_encoder = Transformer(config)
        self.vm_encoder = VMEncoder(config)
        self.vm_decoder = VMDecoder(config)

        self.last_layer = nn.Sigmoid()

        if self.config.use_hscam:
            self.hscam_layer = HSCAMLayer(config)

        self.feature_diff_prod = FeatureDiffAndProd(config)
        self.TF_BottleneckModule = BottleneckModule(config)
        self.VM_BottleneckModule = BottleneckModule(config)

    def forward(self, x):
        x = self.patch_embed(x)
        vm_x, vm_skip_list = self.vm_encoder(x)
        tf_x, tf_skip_list = self.tf_encoder(x)

        tf_skip_list = batch_rearrange(tf_skip_list)
        dp_convs, diffs, prods = self.feature_diff_prod(vm_skip_list, tf_skip_list)

        if self.config.use_hscam:
            dp_convs = self.hscam_layer(dp_convs)

        tf_x = _rearrange(tf_x)
        tf_middle_out = self.TF_BottleneckModule(tf_x)
        vm_middle_out = self.VM_BottleneckModule(vm_x)

        for i in range(len(diffs)):
            diffs[i] = self.last_layer(diffs[i])
            prods[i] = self.last_layer(prods[i])

        out = self.vm_decoder(vm_x, dp_convs)

        return self.last_layer(out), tf_middle_out, vm_middle_out, diffs, prods

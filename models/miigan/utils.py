import torch
import torch.nn as nn
from einops import rearrange
import functools
import torch.nn.functional as F


# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward_wo_sn(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class BottleneckModule(nn.Module):
    def __init__(self, config):
        super(BottleneckModule, self).__init__()
        self.activation = nn.ReLU(inplace=False)
        self.linear = functools.partial(SNLinear, num_svs=1, num_itrs=1, eps=1e-12)
        self.linear_middle = self.linear(config.mamba.embed_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sum(self.activation(x), [2, 3])
        out = self.linear_middle(x)
        out = self.sigmoid(out)
        return out


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False).cuda(0)
        self.norm = norm_layer(4 * dim).cuda(0)

    def forward(self, x):
        B, H, W, C = x.shape
        x.cuda(0)
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class GenPatchEmbed2D(nn.Module):
    """Construct the embeddings from patch
    """

    def __init__(self, config):
        super(GenPatchEmbed2D, self).__init__()
        patch_size = (config.patch_size, config.patch_size)

        # img_size = _pair(img_size)
        # n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = nn.Conv2d(in_channels=config.mamba.input_chans,
                                          out_channels=config.mamba.embed_dims[0],
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.norm = nn.LayerNorm(config.mamba.embed_dims[0])

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.permute(0, 2, 3, 1)  # (B, n_patches^(1/2), n_patches^(1/2), hidden)
        x = self.norm(x)
        return x


class DisPatchEmbed2D(nn.Module):
    """Construct the embeddings from patch
    """

    def __init__(self, config):
        super(DisPatchEmbed2D, self).__init__()
        patch_size = (config.patch_size, config.patch_size)

        # img_size = _pair(img_size)
        # n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = nn.Conv2d(in_channels=config.mamba.input_chans + 1,
                                          out_channels=config.mamba.embed_dims[0],
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.norm = nn.LayerNorm(config.mamba.embed_dims[0])

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.permute(0, 2, 3, 1)  # (B, n_patches^(1/2), n_patches^(1/2), hidden)
        x = self.norm(x)
        return x


def dim_conversion(x):
    x = x.permute(0, 3, 1, 2)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
    x = x.flatten(2)
    x = x.transpose(-1, -2)  # (B, n_patches, hidden)
    return x


def _rearrange(x):
    x = rearrange(x, 'batch (h w) c -> batch h w c', h=int(x.size(1) ** 0.5))
    return x


def batch_rearrange(feature_list):
    re = []
    for feature in feature_list:
        re.append(_rearrange(feature))
    return re

import torch
import math
import torch.nn as nn
from common import *
from einops import rearrange
from network.csa import CSA
from timm.models.layers import DropPath, trunc_normal_
import scipy.io as sio

class CST(nn.Module):
    """SST
     Spatial-Spectral Transformer for Hyperspectral Image Denoising
        Args:
            inp_channels (int, optional): Input channels of HSI. Defaults to 31.
            dim (int, optional): Embedding dimension. Defaults to 90.
            window_size (int, optional): Window size of non-local spatial attention. Defaults to 8.
            depths (list, optional): Number of Transformer block at different layers of network. Defaults to [ 6,6,6,6,6,6].
            num_heads (list, optional): Number of attention heads in different layers. Defaults to [ 6,6,6,6,6,6].
            mlp_ratio (int, optional): Ratio of mlp dim. Defaults to 2.
            qkv_bias (bool, optional): Learnable bias to query, key, value. Defaults to True.
            qk_scale (_type_, optional): The qk scale in non-local spatial attention. Defaults to None. If it is set to None, the embedding dimension is used to calculate the qk scale.
            bias (bool, optional):  Defaults to False.
            drop_path_rate (float, optional):  Stochastic depth rate of drop rate. Defaults to 0.1.
    """

    def __init__(self,
                 inp_channels=31,
                 dim=90,
                 depths=[6, 6, 6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6, 6, 6],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1,
                 scale=4
                 ):
        super(CST, self).__init__()

        self.conv_first = nn.Conv2d(inp_channels, dim, 3, 1, 1)  # shallow featrure extraction
        self.num_layers = depths
        self.layers = nn.ModuleList()
        print("network depth:", len(self.num_layers))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(len(self.num_layers)):
            layer = Cstage(dim=dim,
                              depth=depths[i_layer],
                              num_head=num_heads[i_layer],
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                              bias=bias)
            self.layers.append(layer)

        # self.conv_delasta = nn.Conv2d(dim, inp_channels, 3, 1, 1)  # reconstruction from features
        self.skip_conv = default_conv(inp_channels, dim, 3)
        self.upsample = Upsampler(default_conv, scale, dim)
        self.tail = default_conv(dim, inp_channels, 3)
        self.conv = default_conv(dim,dim,3)

    def forward(self, inp_img, lms):
        f1 = self.conv_first(inp_img)
        # ff = f1.detach().cpu().numpy()
        # outputfile = "f1.mat"
        # sio.savemat(outputfile, {'features':ff})
        # print("save successfully")

        x = f1
        for i in range(len(self.num_layers)):
            x = self.layers[i](x)
        x = self.conv(x + f1)
        # x = self.conv_delasta(x) + inp_img
        x = self.upsample(x)
        x = x + self.skip_conv(lms)
        x = self.tail(x)
        return x


class Cstage(nn.Module):
    def __init__(self,
                 dim=90,
                 split_size=(2,16),
                 depth=6,
                 num_head=6,
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 drop_path=0.1,
                 bias=False):
        super(Cstage, self).__init__()
        self.layers1 = nn.ModuleList()
        self.layers2 = ResAttentionBlock(default_conv, dim, 1, res_scale=0.1)
        self.depth = depth
        for i_layer in range(depth):
            self.layers1.append(CSMA(dim=dim,
                                     input_resolution=(32, 32),
                                     num_heads=num_head,
                                     drop_path=drop_path[i_layer],
                                     split_size=split_size,
                                     shift_size=[0,0] if (i_layer % 2 == 0) else [split_size[0]//2, split_size[1]//2],
                                     mlp_ratio=mlp_ratio,
                                     attn_drop=0,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale, bias=bias))
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x1 = x
        for i in range(self.depth):
            x1 = self.layers1[i](x1)
        x2 = self.layers2(x)
        out = self.conv(x1) + x2
        out = x + out
        return out


class CSE(nn.Module):
    """global spectral attention (CSE)
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """

    def __init__(self, dim, num_heads, bias, k=0.5, sr_ratio=2):
        super(CSE, self).__init__()
        self.num_heads = num_heads
        self.k = int(k * dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.qkv = nn.Conv2d(dim, k*3, kernel_size=1, bias=bias)
        self.sr_ratio = sr_ratio
        self.v = nn.Conv2d(dim, self.k, kernel_size=1, bias=bias)
        self.qk = BSConvU(dim, 2 * self.k, kernel_size=sr_ratio, stride=sr_ratio, padding=0)
        self.project_out = nn.Conv2d(self.k, dim, kernel_size=1, bias=bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        qk = self.qk(x)
        q, k = qk.chunk(2, dim=1)  # b self.k h/s w/s
        v = self.v(x)  # b k h w
        q = q.reshape(b, self.num_heads, self.k // self.num_heads, -1)
        k = k.reshape(b, self.num_heads, self.k // self.num_heads, -1)
        v = v.reshape(b, self.num_heads, self.k // self.num_heads, -1)  # b k h w

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def flops(self, patchresolution):
        flops = 0
        H, W, C = patchresolution
        flops += H * C * W * C
        flops += C * C * H * W
        return flops


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.bsconv = BSConvU(dim, hidden_features*2, kernel_size=3,  stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = self.bsconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class CSMA(nn.Module):
    def __init__(self, dim, input_resolution=[32,32], num_heads=6, drop_path=0.0, split_size=[7, 7], shift_size=[0,0],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, bias=False):
        super(CSMA, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = FeedForward(dim)

        self.attns = CSA(
            dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            split_size=split_size,
            shift_size=shift_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.spectral_attn = CSE(dim, num_heads, bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = self.attns(x, (H,W))

        x = x.view(B, H * W, C)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.spectral_attn(x)  # global spectral attention

        x = x.flatten(2).transpose(1, 2)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.norm2(x).transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2))

        x = x.transpose(1, 2).view(B, C, H, W)
        return x


#
import torch
import numpy as np
from typing import List

from structures import get_equivariant_subspace


# DO NOT CHANGE. Models.MLP inherit from this class
class Model(torch.nn.Module):
    R"""
    Model.
    """
    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        ...


class MLP(Model):
    R"""
    MLP.
    """
    def __init__(self, /, *, size: int, shapes: List[int]) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        buf = []
        shapes = [size * size] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            #
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        x = torch.flatten(x, start_dim=1)
        for (l, linear) in enumerate(self.linears):
            #
            x = linear.forward(x)
            if l < len(self.linears) - 1:
                #
                x = torch.nn.functional.relu(x)
        return x


#
PADDING = 3


class CNN(torch.nn.Module):
    R"""
    CNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, channels: List[int], shapes: List[int],
        kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # Create a list of Conv2D layers and shared max-pooling layer.
        # Input and output channles are given in `channels`.
        # ```
        # buf_conv = []
        # ...
        # self.convs = torch.nn.ModuleList(buf_conv)
        # self.pool = ...
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

        # Create a list of Linear layers.
        # Number of layer neurons are given in `shapes` except for input.
        # ```
        # buf = []
        # ...
        # self.linears = torch.nn.ModuleList(buf)
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = np.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = np.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        # CNN forwarding whose activation functions should all be relu.
        # YOU SHOULD FILL IN THIS FUNCTION
        ...


# =============================================================================
# Q3.c.4: G-Equivariant CNN for orientation prediction 
# =============================================================================

class GEquivariantConv2d(torch.nn.Module):
    """
    Rotation-equivariant patch->patch layer.

    Input:  x  [B, in_channels, H, W]
    Output: y  [B, out_channels, H_out, W_out]
    """

    def __init__(self, in_channels, out_channels, kernel_size_in, kernel_size_out,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_in = kernel_size_in
        self.k_out = kernel_size_out
        self.stride = stride
        self.padding = padding

        # basis: [n_basis, dim_out, dim_in]
        # dim_in = in_channels * k_in^2, dim_out = k_out^2
        basis_np = get_equivariant_subspace(in_channels, kernel_size_in, kernel_size_out)
        self.register_buffer('basis', torch.from_numpy(basis_np).float())

        n_basis = basis_np.shape[0]

        # Separate coefficients per output channel (out_channels is multiplicity)
        # coeffs: [out_channels, n_basis]
        self.coeffs = torch.nn.Parameter(torch.empty(out_channels, n_basis))

        # Scalar bias per output channel, broadcast to all k_out^2 patch entries
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        stdv = 1.0 / np.sqrt(n_basis)
        self.coeffs.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        Args:
            x: input images [batch, 1, 28, 28]

        Returns:
            logits: orientation logits [batch, 4]
        """
        B, C, H, W = x.shape
        dim_in = self.in_channels * self.k_in * self.k_in
        dim_out = self.k_out * self.k_out

        # 1) Extract input patches: [B, dim_in, L]
        cols = torch.nn.functional.unfold(
            x, kernel_size=self.k_in, padding=self.padding, stride=self.stride)
        L = cols.shape[2]
        '''
        Understanding torch.nn.functional.unfold:
        Assume x is [1, 1, 5, 5], kernel_size=3, padding=0, stride=2.

        Input (5x5):
        a b c d e
        f g h i j
        k l m n o
        p q r s t
        u v w x y

        Sliding 3x3 windows with stride 2 produce patches at top-left, top-right, bottom-left,
        bottom-right:

        Patch 1 (top-left):  a b c
                            f g h
                            k l m   -> flattened -> [a,b,c,f,g,h,k,l,m]

        Patch 2 (top-right): c d e
                            h i j
                            m n o   -> flattened -> [c,d,e,h,i,j,m,n,o]

        Patch 3 (bottom-left): k l m
                                p q r
                                u v w   -> flattened -> [k,l,m,p,q,r,u,v,w]

        Patch 4 (bottom-right): m n o
                                r s t
                                w x y   -> flattened -> [m,n,o,r,s,t,w,x,y]

        unfold returns a tensor of shape [1, 9, 4] (1 batch, 9-element vectors, 4 patches). This
        is exactly what the subsequent matrix multiplication expects.
        '''

        H_blocks = (H + 2 * self.padding - self.k_in) // self.stride + 1
        W_blocks = (W + 2 * self.padding - self.k_in) // self.stride + 1

        # 2) Build per-output-channel linear maps from coeffs and basis
        Wmap = ??
    
        # 3) Apply Wmap to each patch. Note that we get a vector as output per patch. This vector is the new patch in the feature map.
        patch_outputs = ??

        if self.bias is not None:
            ??
        
        # 4) Reshape (refolds) output patches back into a feature map
        # fold expects [B, out_channels * dim_out, L]
        patches_flat = patch_outputs.reshape(B, self.out_channels * dim_out, L)

        H_out = (H_blocks - 1) * self.stride + self.k_out
        W_out = (W_blocks - 1) * self.stride + self.k_out

        # torch.nn.functional.fold is the inverse of unfold.
        y = torch.nn.functional.fold(
            patches_flat,
            output_size=(H_out, W_out),
            kernel_size=self.k_out,
            stride=self.stride,
            padding=0)

        return y



class GEquivariantCNN(torch.nn.Module):
    """
    Rotation-equivariant CNN

    Architecture:
        [B, 1, 28, 28]
        -> GEquivariantConv2d(1, 4, k_in=6, k_out=4, s=2)  -> [B, 4, 26, 26]
        -> ELU
        -> GEquivariantConv2d(4, 2, k_in=6, k_out=4, s=2)  -> [B, 2, 24, 24]
        -> ELU
        -> GEquivariantConv2d(2, 1, k_in=6, k_out=2, s=1)  -> [B, 1, 20, 20]
        -> adaptive_avg_pool2d(2)                             -> [B, 1, 2, 2]
        -> flatten                                            -> [B, 4]

    """

    def __init__(self):
        super().__init__()
        self.conv1 = GEquivariantConv2d(
            in_channels=1, out_channels=4,
            kernel_size_in=6, kernel_size_out=4, stride=2)
        self.act1 = torch.nn.ELU()
        self.conv2 = GEquivariantConv2d(
            in_channels=4, out_channels=2,
            kernel_size_in=6, kernel_size_out=4, stride=2)
        self.act2 = torch.nn.ELU()
        self.conv3 = GEquivariantConv2d(
            in_channels=2, out_channels=1,
            kernel_size_in=6, kernel_size_out=2, stride=1)

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = self.act1(self.conv1(x))       # [batch, 4, 26, 26]
        x = self.act2(self.conv2(x))       # [batch, 2, 24, 24]
        x = self.conv3(x)                  # [batch, 1, 20, 20]
        x = torch.nn.functional.adaptive_avg_pool2d(x, 2)  # [batch, 1, 2, 2]
        return x.flatten(start_dim=1)      # [batch, 4]


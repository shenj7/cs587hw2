#
# structures.py - Helper utilities for HW2.
#
# Part 1: Rotation/flip utilities used by main.py (Q3.c.1-3)
# Part 2: Reynolds operator and equivariant basis computation (Q3.c.4)
#

import argparse
import numpy as np
import numpy.typing as npt


# ============================================================================
# Part 1: Rotation and flip utilities
# ============================================================================

def rotate(
    mat: npt.NDArray[np.generic], d: int,
    /,
) -> npt.NDArray[np.generic]:
    R"""
    Rotate a 2D matrix by d * 90 degrees.
    """
    #
    return np.rot90(mat, d)


def flip(
    mat: npt.NDArray[np.generic], d: int,
    /,
) -> npt.NDArray[np.generic]:
    R"""
    Flip a 2D matrix.
    """
    #
    if d == 0:
        #
        return mat
    elif d == 1:
        return np.flip(mat, 1)
    elif d == 2:
        return np.flip(mat, 0)
    elif d == 3:
        return np.flip(mat, (0, 1))
    else:
        #
        raise RuntimeError("Unsupported flipping argument.")


# ============================================================================
# Part 2: Reynolds operator and equivariant basis (Q3.c.4)
# ============================================================================


from functools import lru_cache

# lru_cache caches the results of previous calls to `get_equivariant_subspace`
@lru_cache(maxsize=None)
def get_equivariant_subspace(in_channels, k_in, k_out):
    """
    Compute rotation-equivariant basis for convolution kernels via Reynolds operator.
        W: R^{in_channels * k_in^2} -> R^{k_out^2}

    The representations are:
      - input:  rotate each channel's k_in x k_in patch (channels are trivial)
      - output: rotate the k_out x k_out output patch

    Args:
        in_channels: number of input channels
        k_in: input kernel spatial size
        k_out: output patch spatial size

    Returns:
        basis: [n_basis, dim_out, dim_in]  (dim_out = k_out^2, dim_in = in_channels*k_in^2)
    """
    dim_in = in_channels * k_in * k_in
    dim_out = k_out * k_out

    def transform_in(vec, degree):
        """Rotate every channel's k_in x k_in slice."""
        k = (degree // 90) % 4
        patches = np.asarray(vec).reshape(in_channels, k_in, k_in)
        rotated = np.rot90(patches, k=k, axes=(1, 2))
        return np.ascontiguousarray(rotated).reshape(-1)

    def transform_out(vec, degree):
        """Rotate the k_out x k_out output patch."""
        k = (degree // 90) % 4
        patch = np.asarray(vec).reshape(k_out, k_out)
        rotated = np.rot90(patch, k=k)
        return np.ascontiguousarray(rotated).reshape(-1)

    
    # Build Reynolds operator:
    #   T_bar = (1/|G|) sum_g rho_out(g) kron rho_in(g^{-1})^T
    mats = []
    degrees = [0, 90, 180, 270]
    
    for degree in degrees:
        rho_out = np.stack([transform_out(e, degree) for e in np.eye(dim_out)], axis=1)
        rho_in_inv = np.stack([transform_in(e, -degree) for e in np.eye(dim_in)], axis=1)
        mats.append(np.kron(rho_out, rho_in_inv.T))
    
    T_bar = sum(mats) / len(mats)

    lambda_, V = np.linalg.eig(T_bar)
    
    one_mask = np.isclose(lambda_, 1.0, atol=1e-6)
    
    basis_flat = V[:, one_mask].T.real
    basis = basis_flat.reshape(-1, dim_out, dim_in).astype(np.float32)
    return basis
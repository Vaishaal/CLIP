import torch
import numbers
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from torch import Tensor, Size
from typing import Union, List, Tuple
from torch.nn.modules.normalization import _shape_t


class LayerNormFloat16Support(nn.Module):
    """The same as nn.LayerNorm, but with support for float16 inputs.

    Torchscript doesn't support inheritance, so we have to copy-paste a lot of
    code here.
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...] 
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        """No changes from nn.LayerNorm."""
        super(LayerNormFloat16Support, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """No changes from nn.LayerNorm."""
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Changed to support float16."""
        orig_type = input.dtype
        ret = F.layer_norm(
            input.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return ret.to(orig_type)

    def extra_repr(self) -> str:
        """No changes from nn.LayerNorm."""
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


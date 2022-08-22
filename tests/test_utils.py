import functools
from typing import Optional, List

import pytest
import torch
from torch import Tensor

from torch_pso import OPTIMIZERS


def optimizer_tests(func=None, *, ignore: Optional[List] = None):
    """Decorator function with arguments
    Decorator can be used with or without arguments
    """

    # 1. Decorator arguments are applied to itself as partial arguments
    if func is None:
        return functools.partial(optimizer_tests, ignore=ignore)

    # 2. logic with the arguments

    # 3. Handles the actual decorating
    @pytest.mark.parametrize('optimizer_type', OPTIMIZERS)
    @functools.wraps(func)
    def wrapper(optimizer_type, *args, **kwargs):
        # Write decorator function logic here
        # Before function call
        # ...
        if ignore and optimizer_type.__name__ in ignore:
            # These PSO algorithms converge very slowly on this problem
            pytest.skip()
        return func(optimizer_type, *args, **kwargs)

    return wrapper


def close_to_a_minimum(x: Tensor, minima: List[Tensor], atol: float, rtol: float) -> bool:
    return any(torch.allclose(x, m, atol=atol, rtol=rtol) for m in minima)
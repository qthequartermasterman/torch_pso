import functools
from typing import Optional, List

import pytest
import torch
from torch import Tensor

from torch_pso import OPTIMIZERS


def optimizer_tests(func=None, *, ignore: Optional[List[str]] = None):
    """
    Decorator function with arguments to parameterize test cases for all Optimizers.
    Decorator can be used with or without arguments
    :param func: test function to wrap
    :param ignore: List of strings containing the names of optimizers to skip while parameterizing the test
    """

    # 1. Decorator arguments are applied to itself as partial arguments
    if func is None:
        return functools.partial(optimizer_tests, ignore=ignore)

    # 3. Handles the actual decorating
    @pytest.mark.parametrize('optimizer_type', OPTIMIZERS)
    @functools.wraps(func)
    def wrapper(optimizer_type, *args, **kwargs):
        if ignore and optimizer_type.__name__ in ignore:
            # These PSO algorithms converge very slowly on this problem, so skip them.
            pytest.skip()
        return func(optimizer_type, *args, **kwargs)

    return wrapper


def close_to_a_minimum(x: Tensor, minima: List[Tensor], atol: float, rtol: float) -> bool:
    """
    Determine if the tensor x is close to any of the listed minima.
    :param x: Tensor to test
    :param minima: List of tensors to test against x.
    :param atol: absolute tolerance level. For more details see `torch.allclose`.
    :param rtol: relative tolerance level. For more details see `torch.allclose`.
    :return:
    """
    return any(torch.allclose(x, m, atol=atol, rtol=rtol) for m in minima)
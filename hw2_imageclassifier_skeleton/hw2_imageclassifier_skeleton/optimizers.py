#
import torch
from typing import Iterable, Optional, Callable


class Optimizer(torch.optim.Optimizer):
    r"""
    Optimizer.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        # Save necessary attributes.
        self.lr = lr
        self.weight_decay = weight_decay

        # Super call.
        torch.optim.Optimizer.__init__(self, parameters, dict())

    @torch.no_grad()
    def prev(self, /) -> None:
        r"""
        Operations before compute the gradient.
        PyTorch has design problem of compute Nesterov SGD gradient.
        PyTorch team avoid this problem by using an approximation of Nesterov
        SGD gradient.
        Also, using closure can also solve the problem, but it maybe a bit
        complicated for this homework.
        In our case, function is provided as auxiliary function for simplicity.
        It is called before `.backward()`.
        This function is only used for Nesterov SGD gradient.
        """
        # Do nothing.
        pass

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        #
        ...


class SGD(Optimizer):
    r"""
    SGD.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.
        """
        #
        Optimizer.__init__(self, parameters, lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.
        """
        # Traverse parameters of each groups.
        for group in self.param_groups:
            #
            for parameter in group['params']:
                # Get gradient without weight decaying.
                if parameter.grad is None:
                    #
                    continue
                else:
                    #
                    gradient = parameter.grad

                # Apply weight decay.
                # YOU SHOULD FILL IN THIS FUNCTION
                ...

                # Gradient Decay.
                parameter.data.add_(gradient, alpha=-self.lr)
        return None

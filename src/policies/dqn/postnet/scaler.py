import math
from typing import Literal
import torch
from torch import nn


def clamp_preserve_gradients(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """
    Clamps the values of the tensor into ``[lower, upper]`` but keeps the gradients.
    Args:
        x: The tensor whose values to constrain.
        lower: The lower limit for the values.
        upper: The upper limit for the values.
    Returns:
        The clamped tensor.
    """
    return x + (x.clamp(min=lower, max=upper) - x).detach()


CertaintyBudget = Literal["constant", "exp-half", "exp", "normal"]
"""
The certainty budget to distribute in the latent space of dimension ``H``:
- ``constant``: A certainty budget of 1, independent of the latent space's dimension.
- ``exp-half``: A certainty budget of ``exp(0.5 * H)``.
- ``exp``: A certainty budget of ``exp(H)``.
- ``normal``: A certainty budget that causes a multivariate normal distribution to yield the same
  probability at the origin at any dimension: ``exp(0.5 * log(4 * pi) * H)``.
"""


class EvidenceScaler(nn.Module):
    """
    Scaler for the evidence to distribute a certainty budget other than one in the latent space.
    """

    def __init__(self, dim: int, budget: CertaintyBudget):
        """
        Args:
            dim: The dimension of the latent space.
            budget: The budget to use.
        """
        super().__init__()
        if budget == "exp-half":
            self.log_scale = 0.5 * dim
        elif budget == "exp":
            self.log_scale = dim
        elif budget == "normal":
            self.log_scale = 0.5 * math.log(4 * math.pi) * dim
        else:
            self.log_scale = 0

    def forward(self, log_evidence: torch.Tensor) -> torch.Tensor:
        """
        Scales the evidence in the log space according to the certainty budget.
        Args:
            log_evidence: The log-evidence to scale.
        Returns:
            The scaled and clamped evidence in the log-space.
        """
        return clamp_preserve_gradients(log_evidence + self.log_scale, lower=-100.0, upper=100.0)

from typing import List
import torch
from torch import nn
import src.policies.dqn.postnet.distributions as D
from ._base import Output


def chunk_squeeze_last(x: torch.Tensor) -> List[torch.Tensor]:
    """
    Splits the provided tensor into individual elements along the last dimension and returns the
    items with the last dimension squeezed.
    Args:
        x: The tensor to chunk.
    Returns:
        The squeezed chunks.
    """
    chunks = x.chunk(x.size(-1), dim=-1)
    return [c.squeeze(-1) for c in chunks]


class NormalOutput(Output):
    """
    Normal output with Normal Gamma prior. The prior yields a mean of 0 and a scale of 10.
    """

    def __init__(self, dim: int, prior_mean: float = 0., prior_scale: float = 10., prior_evidence: float = 1.):
        """
        Args:
            dim: The dimension of the latent space.
        """
        super().__init__()
        self.linear = nn.Linear(dim, 2)
        #self.prior = D.NormalGammaPrior(mean=0, scale=10, evidence=1)
        self.prior = D.NormalGammaPrior(mean=prior_mean, scale=prior_scale, evidence=prior_evidence)

    def forward(self, x: torch.Tensor) -> D.Likelihood:
        z = self.linear.forward(x)
        loc, log_precision = chunk_squeeze_last(z)
        return D.Normal(loc, log_precision.exp() + 1e-20)  # Fix numerical issues, otherwise we obtain precision=0

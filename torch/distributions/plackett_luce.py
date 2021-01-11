from typing import Optional

import torch
from torch.distributions import Distribution, constraints


class PlackettLuce(Distribution):
    r"""
    Creates a Plackett-Luce distribution over permutations, parameterized by :attr: `logits`.

    The Plackett-Luce distribution defines a probability distribution over permutations by assigning a score `a_i` to
    each element, and repeatedly choosing the next element by sampling from the remaining elements with a probability
    proportional to their score.

    If :attr:`logits` is 1-dimensional with length-`K`, each element is the log-score of the object at that index.

    If :attr:`logits` is N-dimensional, the first N-1 dimensions are treated as a batch of log-score vectors.

    This distribution supports batched operations with permutations of different sizes, by using the :attr:
    `permutation_sizes` attribute to specify the permutation size of each score vector in the batch.  If the
    permutation_size is `N` for a given index of the batch, the first `N` entries of the resulting sample will be a
    permutation of the number `1` through `N`, while the remainder have unspecified values.

    Example::

        >>> m = PlackettLuce(torch.tensor([[0, 1, -1], [0, 1, 2]]), torch.tensor([3, 2], dtype=torch.int64))
        >>> m.sample()
        tensor([[ 1,  0,  2],
                [ 0,  1,  2]])

    Args:
        logits (Tensor): The log of the Plackett-Luce distribution scores `a_i`.
        permutation_sizes (Tensor): Optional sizes of the permutations sampled by the distribution.  Should match the
                                    shape of the logits, with the last dimension stripped.
    """
    arg_constraints = {'logits': constraints.real}
    support = constraints.integer_interval(-1, torch.iinfo(torch.int64).max)

    def __init__(self, logits: torch.Tensor, permutation_sizes: Optional[torch.Tensor] = None, validate_args=None):
        batch_shape = logits.shape[:-1]
        max_size = logits.shape[-1]

        if permutation_sizes is None:
            permutation_sizes = torch.full(batch_shape, max_size, dtype=torch.int64, device=logits.device)
        else:
            permutation_sizes = permutation_sizes.expand(batch_shape)

        if validate_args:
            if (logits < -1e30).any():
                raise ValueError("Plackett-Luce implementation cannot handle logits less than -1e30")
        self.logits = logits
        self.permutation_sizes = permutation_sizes

        # Mask is true for invalid indices
        self.mask: torch.Tensor = torch.zeros(
            *batch_shape, max_size + 1, device=logits.device
        ).scatter(-1, permutation_sizes.unsqueeze(-1), 1)[..., :-1].cumsum(dim=-1).bool()

        event_shape = torch.Size((max_size,))
        super(PlackettLuce, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            expanded = self.logits.expand(*sample_shape, *[-1] * len(self.logits.shape))
            gumbel_noise = - torch.log(-torch.log(torch.rand_like(expanded)))
            scores = torch.where(self.mask, -1e35, expanded + gumbel_noise)
            sorted_scores, indices = torch.sort(scores, dim=-1, descending=True)
            return indices.masked_fill(self.mask, -1).detach()

    def log_prob(self, value: torch.Tensor):
        if self._validate_args:
            self._validate_sample(value)
        return _plackett_luce_log_prob(self.logits, self.permutation_sizes, self.mask, value)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PlackettLuce, _instance)
        batch_shape = torch.Size(batch_shape)
        logits_shape = batch_shape + (self.logits.shape[-1],)
        new.logits = self.logits.expand(logits_shape)
        new.mask = self.mask.expand(logits_shape)
        new.permutation_sizes = self.permutation_sizes.expand(batch_shape)
        super(PlackettLuce, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _validate_sample(self, value: torch.Tensor):
        super()._validate_sample(value)
        max_int64 = torch.iinfo(torch.int64).max
        if (value.masked_fill(self.mask, max_int64).sort(-1).values
                != torch.arange(0, value.shape[-1], dtype=torch.int64).masked_fill(self.mask, max_int64)).any():
            raise ValueError("Not a valid permutation or batch of permutations.")


@torch.jit.script_if_tracing
def _plackett_luce_log_prob(logits, permutation_sizes, mask, value):
    value = value.masked_fill(mask, 0)
    logits = logits.masked_fill(mask, -1e35).expand(value.shape)
    log_probs = torch.zeros(value.shape[:-1], device=value.device)
    for i in range(int(permutation_sizes.max())):
        log_probs += torch.where(mask[..., i],
                                 0.0,
                                 logits.log_softmax(dim=-1).gather(-1, value[..., i:i + 1]).squeeze(-1),
                                 )
        logits = logits.scatter(-1, value[..., i:i + 1], -1e35)
    return log_probs

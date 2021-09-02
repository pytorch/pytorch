import random
import collections
from typing import (
    Callable,
    Counter,
    Dict,
    Iterator,
    Optional,
    TypeVar,
)

from torch.utils.data import IterDataPipe

T = TypeVar("T")
U = TypeVar("U")


class UnderSampler(IterDataPipe[T]):
    r""":class:`UnderSampler`.

    Iterable datapipe wrapper for under-sampling.

    Args:
        datapipe: Iterable datapipe to undersample from.
        row_to_label: Function called over each item from datapipe to generate
            label/class.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        datapipe: IterDataPipe[T],
        row_to_label: Callable[[T], U],
        seed: Optional[int] = None,
    ) -> None:
        self.datapipe = datapipe
        self.row_to_label = row_to_label
        self.seed = seed
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError


class DistributionUnderSamplerIterDataPipe(UnderSampler[T]):
    r""":class:`DistributionUnderSamplerIterDataPipe`.

    Iterable datapipe wrapper for under-sampling if a desired output distribution of
    labels/classes is known. This method is based on rejection sampling.

    Args:
        datapipe: Iterable datapipe to undersample from.
        row_to_label: Function called over each item from datapipe to generate
            label/class.
        output_dist: The desired label/class distribution. The keys are the classes
            while the values are the desired class percentages. The values,
            however, do not have to be normalized to sum up to 1.
        input_dist: Optional distribution describing label/class distribution of the
            input. The keys are the classes while the values are the class
            percentages. The values, however, do not have to be normalized to sum up
            to 1. If not known, then :class:`DistributionUnderSamplerIterDataPipe` will
            keep a running estimate of the distribution as it processes datapipe.
            If known, then :class:`DistributionUnderSamplerIterDataPipe` will not update
            the distribution as it processes datapipe.
        seed: Random seed for reproducibility.

    References:
        - https://www.wikiwand.com/en/Rejection_sampling

    NOTE: This class is adapted from https://github.com/MaxHalford/pytorch-resample.
    """

    def __init__(
        self,
        datapipe: IterDataPipe[T],
        row_to_label: Callable[[T], U],
        output_dist: Dict[U, float],
        input_dist: Optional[Dict[U, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if any(v < 0 for v in output_dist.values()):
            raise ValueError("Only non-negative values are allowed in output_dist.")
        if input_dist:
            if any(v <= 0 for v in input_dist.values()):
                raise ValueError("Only positive values are allowed in input_dist.")
            if not (output_dist.keys() <= input_dist.keys()):
                raise ValueError(
                    "All keys in output_dist must be present in input_dist."
                )

        super().__init__(datapipe, row_to_label, seed=seed)
        self.input_dist: Counter[U] = collections.Counter(input_dist)
        self.output_dist: Counter[U] = collections.Counter(output_dist)
        self._update_input_dist: bool = not bool(input_dist)
        # The pivot represents the class for which no undersampling is performed.
        self._pivot: Optional[U] = None

    def __iter__(self) -> Iterator[T]:
        for row in self.datapipe:
            # To ease notation
            f = self.output_dist
            g = self.input_dist

            y = self.row_to_label(row)
            if self._update_input_dist:
                g[y] += 1

            # Determine the sampling ratio
            if self._pivot is None or self._update_input_dist:
                self._pivot = max(g.keys(), key=lambda y: f[y] / g[y])
            numerator = f[y] * g[self._pivot]
            denominator = f[self._pivot] * g[y]
            ratio = (numerator / denominator) if denominator > 0 else 0

            if self.rng.random() < ratio:
                yield row


class ProportionUnderSamplerIterDataPipe(UnderSampler[T]):
    r""":class:`ProportionUnderSamplerIterDataPipe`.

    Iterable datapipe wrapper for under-sampling if it is known how much to undersample
    each label/class.

    Args:
        datapipe: Iterable datapipe to undersample from.
        row_to_label: Function called over each item from datapipe to generate
            label/class.
        proportions: How much to undersample each label/class. The keys are the classes
            while the values indicate what proportion of the rows of a specific class
            should be kept. Example: a proportion of 0.3 for class c indicates that
            30% of rows from datapipe whose label is c should be kept.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        datapipe: IterDataPipe[T],
        row_to_label: Callable[[T], U],
        proportions: Dict[U, float],
        seed: Optional[int] = None,
    ) -> None:
        if any(p < 0 or p > 1 for p in proportions.values()):
            raise ValueError("All proportions must be within 0 and 1.")
        super().__init__(datapipe, row_to_label, seed=seed)
        self.proportions = proportions

    def __iter__(self) -> Iterator[T]:
        for row in self.datapipe:
            if self.rng.random() < self.proportions[self.row_to_label(row)]:
                yield row

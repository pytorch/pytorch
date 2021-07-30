import dataclasses
import typing

@dataclasses.dataclass(init=True, repr=False)
class Measurement:
    """The result of a Timer measurement.

    NOTE: This is a placeholder. The full (existing) Measurement class will
          be ported in a later PR
    """
    number_per_run: int
    raw_times: typing.List[float]

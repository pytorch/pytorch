import collections
from dataclasses import fields
from enum import Enum


class ReductionHint(Enum):
    INNER = 0
    OUTER = 1
    OUTER_TINY = 2
    DEFAULT = 3


class TileHint(Enum):
    SQUARE = 0
    DEFAULT = 1


# Attempt to import AttrsDescriptor from Triton
try:
    from triton.compiler.compiler import AttrsDescriptor

    attrs_descriptor_available = True
    # Determine if 'ids_of_folded_args' is a valid field for AttrsDescriptor
    attr_desc_fields = {f.name for f in fields(AttrsDescriptor)}
    ids_of_folded_args_available = "ids_of_folded_args" in attr_desc_fields
    divisible_by_8_available = "divisible_by_8" in attr_desc_fields
except ImportError:
    attrs_descriptor_available = False

# Define `instance_descriptor` function with clear conditional handling
if attrs_descriptor_available:

    def instance_descriptor(
        divisible_by_16=None,
        equal_to_1=None,
        ids_of_folded_args=None,
        divisible_by_8=None,
    ):
        # Prepare the arguments for AttrsDescriptor
        kwargs = {
            "divisible_by_16": divisible_by_16,
            "equal_to_1": equal_to_1,
        }

        # Conditionally add 'ids_of_folded_args' if it's available in AttrsDescriptor
        if ids_of_folded_args_available:
            kwargs["ids_of_folded_args"] = ids_of_folded_args
        if divisible_by_8_available:
            kwargs["divisible_by_8"] = divisible_by_8

        # Instantiate AttrsDescriptor with the prepared arguments
        return AttrsDescriptor(**kwargs)

else:
    # Define a namedtuple as a fallback when AttrsDescriptor is not available
    instance_descriptor = collections.namedtuple(  # type: ignore[no-redef]
        "instance_descriptor",
        ["divisible_by_16", "equal_to_1", "ids_of_folded_args", "divisible_by_8"],
        defaults=[tuple(), tuple(), tuple(), tuple()],
    )

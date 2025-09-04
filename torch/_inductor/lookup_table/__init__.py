"""
Template lookup table system for PyTorch Inductor.

This package provides functionality for:
- Loading pre-configured template choices from lookup tables
- Recording autotuning results for building new lookup tables
- Managing template configurations and choices

The recorder module is imported to automatically register the feedback function
when the lookup_table package is used.
"""

# Import recorder to auto-register feedback function
from . import recorder
from .choices import LookupTableChoices
from .core import lookup_template_configs, make_lookup_key


__all__ = [
    "LookupTableChoices",
    "lookup_template_configs",
    "make_lookup_key",
    "recorder",
]

import unittest
from collections.abc import Sequence
from typing import Optional, Literal

import torch

from .common_utils import MACOS_VERSION
from .opinfo.core import DecorateInfo, OpInfo

class MPSSkipInfo:
    def __init__(
        self,
        *args: Literal["output_match", "error_inputs"],
        test_class: Optional[str] = None,
        variant: Optional[str] = None,
        dtypes: Optional[Union[torch.dtype, list[torch.dtype]]] = None,
        skip: Callable = unittest.expectedFailure,
        skip_msg: str = "Skipped!",
        upper: Optional[float] = None,
        lower: Optional[float] = None,
    ):
        """Basic struct for tracking MPS OpInfo xfails
        args: String names of test(s) to apply this xfail info to
        test_class: Test class, e.g. 'TestCommon' etc.
        variant: Variant name. Set to empty str ("") to explicitly specify the non-variant case
        If set to None, will instead apply to all variants of the test
        dtypes: If none specified, xfails all dtype variants
        skip: Type of decorator to add [expectedFailure, skipTest, xfailUnimplementedOpMPS, xfailUnimplementedDtypeMPS]
        upper: Upper bound MacOS version this xfail applies to (exclusive)
        lower: Lower bound MacOS version this xfail applies to (inclusive)
        """
        self.tests: list[str] = []
        for arg in args:
            self.tests.append(arg)
        self.test_class = test_class
        self.variant = variant
        if type(dtypes) is list:
            self.dtypes = dtypes
        else:
            self.dtypes = [dtypes]
        self.skip = skip
        self.skip_msg = skip_msg
        self.upper = upper
        self.lower = lower

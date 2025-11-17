# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MaskBuffer:
    data: Optional[torch.Tensor] = None
    # refcount allows shared usage of the MaskBuffer, as long as all users have the same data
    refcount: int = 0

    def materialize_mask(self, mask):
        if self.refcount == 0:
            self.data = mask
        else:
            assert self.data is not None
            if not torch.equal(self.data, mask):
                raise RuntimeError(
                    "MaskBuffer has been materialized with conflicting data"
                )
        self.refcount += 1

    def release_mask(self):
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")
        self.refcount -= 1
        if self.refcount == 0:
            self.data = None

    def apply_mask(self, tensor):
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")

        # NOTE: MaskPartial is being used by the embedding op and the gather op.
        # For gather, the mask has the same dimension as the output tensor, whereas
        # the output of the embedding op has an additional dimension compare to the input,
        # hence the output masking logic below having two different cases.
        if tensor.ndim == self.data.ndim:
            tensor[self.data] = 0.0
        else:
            tensor[self.data, :] = 0.0

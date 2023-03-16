from __future__ import annotations

from typing import Any, Sequence

import torch
import torch._ops
from torch.onnx import errors


class UnsupportedCallFunctionError(errors.OnnxExporterError):
    """Raised when unsupported call_function is encountered."""

    targets: Sequence[Any]
    exporter_keys: Sequence[str]

    def __init__(self, targets: Sequence[Any], exporter_keys: Sequence[str]):
        self.targets = targets
        self.exporter_keys = exporter_keys

        targets_msg = ""
        exporter_keys_msg = ""

        if len(targets) > 0:
            str_targets = []
            for target in targets:
                if isinstance(target, torch._ops.OpOverload):
                    str_targets.append(target.name())
                else:
                    str_targets.append(target)
            targets_msg = f"[{', '.join(str_targets)}]. "
        if len(exporter_keys) > 0:
            exporter_keys_msg = f"Cannot find function for {exporter_keys}. "

        super().__init__(targets_msg + exporter_keys_msg)

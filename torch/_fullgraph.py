import dataclasses
import logging
import os
from typing import Dict, List, Optional

import torch
import torch._inductor.package


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _Compilation:
    paths: List[List[str]]


class _Package:
    def __init__(
        self,
        *,
        path: Optional[str] = None,
    ):
        self.path = path or os.path.expandvars("/tmp/torchinductor_$USER/model.pt2")
        if os.path.exists(self.path) and os.path.isdir(self.path):
            raise RuntimeError(
                f"File {self.path} already exists as a directory. Please specify a file path."
            )
        self._compilations: Dict[str, _Compilation] = {}

    def add_aoti(self, name: str, path: List[str]) -> None:
        if name not in self._compilations:
            self._compilations[name] = _Compilation([])
        else:
            raise RuntimeError("Recompilation NYI")
        self._compilations[name].paths.append(path)

    def num_models(self) -> int:
        return len(self._compilations)

    def num_compilations(self, name: str) -> int:
        return len(self._compilations[name].paths)

    def _save(self) -> None:
        if len(self._compilations) == 0:
            logger.warning("No compiled models found for packaging.")
        else:
            torch._inductor.package.package_aoti(
                self.path, {name: c.paths[0] for name, c in self._compilations.items()}
            )

        # TODO self._compilations.clear()

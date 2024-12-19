import dataclasses
import functools
import inspect
import os
import weakref
from typing import Any, Dict, Optional

import torch

import torch._inductor.package


@dataclasses.dataclass
class _Compilation:
    name: str
    model: Any
    path: str
    version: int


class _Package:
    def __init__(
        self,
        *,
        mode: str = "package",
        path: Optional[str] = None,
        frontend: str = "aot_autograd",
    ):
        self.mode = mode
        self.path = path or os.path.expandvars("/tmp/torchinductor_$USER/model.pt2")
        if os.path.exists(self.path) and os.path.isdir(self.path):
            raise RuntimeError(
                f"File {self.path} already exists as a directory. Please specify a file path."
            )
        self.frontend = frontend
        self._compilations: Dict[str, _Compilation] = {}

    def __call__(self, model: Any, name: str, version: int) -> Any:
        assert callable(model)
        assert isinstance(name, str)
        assert isinstance(version, int)

        @functools.wraps(model)
        def _(*args, **kwargs) -> Any:
            if name in self._compilations:
                # Making sure for each compilation name, it's only coming from a single torch.compile call.
                if self._compilations[name].model is not model:
                    raise RuntimeError(
                        f"Multiple compilations to different objects are found with the same model name '{name}'. "
                        + "Please specify a unique name for each compilation with torch.compile."
                    )

                if self._compilations[name].version != version:
                    raise RuntimeError(
                        f"Multiple compilations to the same model name '{name}' are not supported. "
                        + "Please instead specify a unique name for each compilation with torch.compile."
                    )

            if self.mode == "package":
                if isinstance(model, torch.nn.Module):
                    module = model
                else:

                    class Module(torch.nn.Module):
                        def __init__(self) -> None:
                            super().__init__()
                            self.model = model

                        def forward(self, *args, **kwargs) -> Any:
                            return self.model(*args, **kwargs)

                    module = Module()

                if self.frontend == "aot_autograd":
                    strict = False
                elif self.frontend == "dynamo":
                    strict = True
                else:
                    raise RuntimeError(f"Unknown frontend {self.frontend}")
                ep = torch.export.export(module, args, kwargs, strict=strict)

                args, kwargs = ep.example_inputs
                options = {
                    "aot_inductor.package": True,
                }
                path = torch._inductor.aot_compile(
                    ep.module(),  # type: ignore[arg-type]
                    args,
                    kwargs,
                    options=options,
                )
                assert isinstance(path, str)
                self._compilations[name] = _Compilation(
                    name=name,
                    model=model,
                    path=path,
                    version=version,
                )

                runner = model
            elif self.mode == "load":
                # TODO Cache the loaded packages.
                runner = torch._inductor.package.load_package(self.path, name)
            else:
                raise RuntimeError(
                    f"Unknown mode {self.mode}. Supported options: package, load"
                )

            return runner(*args, **kwargs)

        return _

    def finalize(self) -> None:
        torch._inductor.package.package_aoti(
            self.path, {c.name: c.path for c in self._compilations.values()}
        )
        self._compilations.clear()


_optimized_versions: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _bump_optimized_version(obj: Any) -> int:
    version = _optimized_versions.get(obj, 0)
    _optimized_versions[obj] = version + 1
    return version


def _get_model_name(model: Any) -> str:
    if isinstance(model, torch.nn.Module):
        name = model.__class__.__qualname__ + ".forward"
    elif inspect.ismethod(model) or inspect.isfunction(model):
        name = model.__qualname__
    else:
        raise RuntimeError(f"Unknown model type {type(model)}")

    module_name = model.__module__
    return module_name + "." + name

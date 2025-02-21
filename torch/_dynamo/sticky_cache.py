"""
This module provides the infrastructure for creating and managing sticky cache
for torch.compile. We mainly have two abstractions here:
  - Precompile: Compiled artifacts related to one particular recompile.
  - StickyCache: Overarching data structure for store and lookup a list of precompiles.

This is different from the typical global caching system in the sense that sticky cache is
always saved/loaded via user API calls. This means the caching behavior is always under
user control explicitly so that a stronger guarantee can be provided about cache hit for a
specific compiled model. Users can load the sticky cache from a different process or even
host but cautions should be taken that sticky cache will only check a subset of the original
Dynamo guards so there might be soundness problems.
"""

import contextlib
import dataclasses
import functools
import glob
import logging
import os
import pickle
import shutil
import types
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch._inductor.package
from torch._inductor.codecache import extract_tensor_metadata_for_cache_key
from torch._inductor.output_code import CompiledAOTI
from torch._subclasses.fake_tensor import TensorMetadata

from .bytecode_transformation import get_code_keys


logger = logging.getLogger(__name__)


# Dict type is not hashable, using values instead.
@functools.lru_cache(maxsize=1)
def _extract_inputs_metadata_cached(
    args: Tuple[Any, ...], kwargs: Tuple[Any, ...]
) -> List[TensorMetadata]:
    assert all(type(x) == torch.Tensor for x in args)
    assert all(type(x) == torch.Tensor for x in kwargs)
    return [extract_tensor_metadata_for_cache_key(t) for t in args + kwargs]


def _extract_inputs_metadata(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> List[TensorMetadata]:
    return _extract_inputs_metadata_cached(args, tuple(kwargs.values()))


class _Precompile:
    """
    A precompile contains all the serializable information associated with a single
    compilation in torch.compile(). To restore an execution of compiled code, we will
    need to serialize the following (not exhaustive):
      - AOTI compiled code and kernels for forward and backward graph.
      - AOTAutograd wrappers for things like input mutation.
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards for cache keys.
    """

    def __init__(self) -> None:
        self.inputs_metadata: Optional[List[TensorMetadata]] = None
        self.aoti: Optional[CompiledAOTI] = None
        self.dynamo_code: Optional[types.CodeType] = None

    def add_aoti(self, aoti: CompiledAOTI) -> None:
        assert self.aoti is None
        self.aoti = aoti

    def add_inputs_metadata(self, inputs_metadata: List[TensorMetadata]) -> None:
        assert self.inputs_metadata is None
        self.inputs_metadata = inputs_metadata

    def add_dynamo_code(self, dynamo_code: types.CodeType) -> None:
        assert self.dynamo_code is None
        self.dynamo_code = dynamo_code

    def match_inputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
        # TODO use dynamo guards when it's ready
        assert self.inputs_metadata is not None
        for a, b in zip(self.inputs_metadata, _extract_inputs_metadata(args, kwargs)):
            if a != b:
                return False
        return True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self.dynamo_code is not None
        name = next(
            n for n in self.dynamo_code.co_names if n.startswith("__compiled_fn")
        )
        return types.FunctionType(self.dynamo_code, globals={name: self.aoti})(
            *args, **kwargs
        )


@dataclasses.dataclass(frozen=True)
class SerializedCode:
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: Tuple[Any, ...]
    co_names: Tuple[str, ...]
    co_varnames: Tuple[str, ...]
    co_filename: str
    co_name: str
    co_qualname: Optional[str]
    co_firstlineno: int
    co_linetable: bytes
    co_exceptiontable: bytes
    co_cellvars: Tuple[str, ...]
    co_freevars: Tuple[str, ...]


def _load_precompile(load_path: str, index: int) -> _Precompile:
    precompile = _Precompile()
    aoti = os.path.join(load_path, f"{index}.pt2")
    model = torch._inductor.package.load_package(aoti, "forward")

    def current_callable(args: Tuple[Any, ...]) -> Any:
        return model.loader.run(list(args))  # type: ignore[attr-defined]

    precompile.add_aoti(CompiledAOTI(aoti, current_callable))

    stripped = aoti.rsplit(".", 1)[0]
    with open(stripped + ".inputs_metadata", "rb") as f:
        precompile.add_inputs_metadata(pickle.load(f))
    with open(stripped + ".dynamo_code", "rb") as f:
        serialized_code = pickle.load(f)
    precompile.add_dynamo_code(
        types.CodeType(*[getattr(serialized_code, key) for key in get_code_keys()])
    )
    return precompile


def _save_precompile(save_path: str, index: int, precompile: _Precompile) -> None:
    assert precompile.aoti is not None
    src = precompile.aoti.filename
    dst = os.path.join(save_path, f"{index}.pt2")
    if isinstance(src, str) and src.endswith(".pt2"):
        # Already packed aoti file.
        shutil.copy(src, dst)
    else:
        # Unpacked aoti files.
        assert isinstance(src, list)
        torch._inductor.package.package_aoti(dst, {"forward": src})
        with open(os.path.join(save_path, f"{index}.inputs_metadata"), "wb") as f:
            pickle.dump(precompile.inputs_metadata, f)
        serialized_code = SerializedCode(
            **{key: getattr(precompile.dynamo_code, key) for key in get_code_keys()}
        )
        with open(os.path.join(save_path, f"{index}.dynamo_code"), "wb") as f:
            assert (
                len(
                    [
                        name
                        for name in serialized_code.co_names
                        if name.startswith("__compiled_fn")
                    ]
                )
                == 1
            )
            pickle.dump(serialized_code, f)


class _StickyCache:
    """
    The main entry point of sticky cache system. This data structure should be created
    per torch.compile() call and propagated through the layers to collect compiled
    artifacts from Dynamo, AOTAutograd and Inductor. This essentially maintains a
    list of (guards, compiled code) which will be looked up in order when a set of
    new inputs are passed to compiled object.
    """

    def __init__(self, path: str):
        self.path = path
        if self.path.endswith(".pt2"):
            self.unimplemented("single file package")
        self._precompiles: List[_Precompile] = []
        self._current_precompile: Optional[_Precompile] = None

    @property
    def current_precompile(self) -> _Precompile:
        """
        Used to access the current precompile object within different compilation
        phases.
        """
        assert self._current_precompile is not None
        return self._current_precompile

    @contextlib.contextmanager
    def precompile_context(self) -> Generator[None, None, None]:
        """
        Set up the current precompile context, should be tied to a full recompilation
        cycle.
        """
        assert self._current_precompile is None
        precompile = _Precompile()
        self._current_precompile = precompile
        try:
            yield
        finally:
            self._precompiles.append(precompile)
            self._current_precompile = None

    def unimplemented(self, msg: str) -> None:
        raise NotImplementedError(
            f"Feature not implemented yet for sticky cache: {msg}."
        )

    def save(self, *, prefix: Optional[str] = None) -> None:
        """
        Implementation of torch.compile().save_stikcy_cache().
        """
        assert self._current_precompile is None
        prefix = prefix or ""
        path = os.path.join(prefix, self.path)
        if len(self._precompiles) == 0:
            logger.warning("No compiled models found for sticky cache.")
        else:
            # TODO Inductor packaging currently doesn't support things like metadata read/write,
            #      to unblock we will have a custom directory for now.
            os.makedirs(path)
            for i, precompile in enumerate(self._precompiles):
                _save_precompile(path, i, precompile)

    def load(self, *, prefix: Optional[str] = None) -> None:
        """
        Implementation of torch.compile().load_stikcy_cache().
        """
        assert self._current_precompile is None
        prefix = prefix or ""
        path = os.path.join(prefix, self.path)
        if not os.path.exists(path) or not os.path.isdir(path):
            raise RuntimeError(f"Sticky cache path '{path}' doesn't exist.")
        aoti_artifacts = glob.glob(os.path.join(path, "*.pt2"))
        precompiles: List[_Precompile] = [_Precompile()] * len(aoti_artifacts)
        for i in range(len(aoti_artifacts)):
            precompiles[i] = _load_precompile(path, i)
        self._precompiles = precompiles

    def lookup(
        self, args: Tuple[Any], kwargs: Dict[str, Any]
    ) -> Optional[Callable[..., Any]]:
        if any(type(x) != torch.Tensor for x in args) or any(
            type(x) != torch.Tensor for x in kwargs.values()
        ):
            self.unimplemented("structured inputs")

        for precompile in self._precompiles:
            if precompile.match_inputs(args, kwargs):

                def _fn(*args: Any, **kwargs: Any) -> Any:
                    return precompile(*args, **kwargs)

                return _fn
        return None

    def reset(self, state: Optional[List[_Precompile]] = None) -> Any:
        assert self._current_precompile is None
        _precompiles = self._precompiles
        self._precompiles = state or []
        assert all(isinstance(p, _Precompile) for p in self._precompiles)
        return _precompiles

import contextlib
import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from torch._library.custom_ops import _maybe_get_opdef


log = logging.getLogger(__name__)


class MissingOpProfile(RuntimeError):
    """
    This is raised when we don't have an operator profile available for the
    given inputs.
    """


@dataclass(frozen=True)
class TensorMetadata:
    rank: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout

    @staticmethod
    def maybe_from_tensor(t: Any) -> Optional["TensorMetadata"]:
        if not isinstance(t, torch.Tensor):
            return None
        return TensorMetadata(t.dim(), t.dtype, t.device, t.layout)


@dataclass(frozen=True)
class OpProfile:
    args_profile: tuple[Optional[TensorMetadata]]
    out_profile: Union[TensorMetadata, tuple[TensorMetadata]]


def _generate_fake_kernel(op_name: str, op_profile: set[OpProfile]) -> Callable:
    def _match_args(args_profile: tuple[Optional[TensorMetadata]], args: Any) -> bool:
        return all(
            TensorMetadata.maybe_from_tensor(arg) == args_profile[i]
            for i, arg in enumerate(args)
        )

    def _generate_res(
        out_profile: Union[TensorMetadata, tuple[TensorMetadata]],
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        ctx = torch.library.get_ctx()

        def _generate_tensor_out(t: TensorMetadata) -> torch.Tensor:
            fake_shape = [ctx.new_dynamic_size() for _ in range(t.rank)]
            fake_strides = [-1] * t.rank
            expected = 1
            fake_stride = expected
            for i in range(t.rank):
                fake_strides[i] = fake_stride  # type: ignore[assignment]
                fake_stride = fake_stride * fake_shape[i]  # type: ignore[assignment]

            return torch.empty_strided(
                fake_shape,
                fake_strides,
                device=t.device,
                dtype=t.dtype,
                layout=t.layout,
            )

        if isinstance(out_profile, TensorMetadata):
            return _generate_tensor_out(out_profile)
        else:
            return [_generate_tensor_out(t) for t in out_profile]

    def _fake_kernel(*args, **kwargs):  # type: ignore[no-untyped-def]
        for profile in op_profile:
            if _match_args(profile.args_profile, (*args, *kwargs.values())):
                return _generate_res(profile.out_profile)

        raise MissingOpProfile(
            f"No fake kernel was found for {op_name}, and although we have "
            "previously registered some profiles to generate a fake kernel, "
            f"no profiles match the given inputs: {args, kwargs}."
        )

    return _fake_kernel


@contextlib.contextmanager
def register_fake_profile(op_profiles: dict[str, set[OpProfile]]) -> Generator:
    """
    Registers a fake kernel based on the given operator profiles. This fake
    kernel registration will override any existing fake kernel registrations.

    The input is a dictionary mapping operator names to a set of operator
    profiles, which we will use to generate fake kernels. The operator profiles
    are a record of the input and output tensor metadata. Based on this
    information we will match a given input to the recorded profile, and return
    an output with the same metadata as in the recorded profile. If a profile
    doesn't exist then an exception will be thrown.

    Args:
        op_profiles (dict[str, set[OpProfile]]): A dictionary mapping operator
            name to a set of operator profiles from which we will generate fake
            kernels.
    """

    libs: list[torch.library.Library] = []
    # Stores old fake impls from custom ops declared through @custom_op
    old_fake_impls: dict[str, Callable] = {}
    for op_name, profiles in op_profiles.items():
        log.warning(
            "Registering fake profile for %s. This will override any existing "
            "fake kernel registration.",
            op_name,
        )

        op_name_split = op_name.split(".")
        namespace, op_name_str = op_name_split[0], op_name_split[1]
        op_str = f"{namespace}::{op_name_str}"

        fake_kernel = _generate_fake_kernel(op_str, profiles)

        if opdef := _maybe_get_opdef(op_str):
            # If the op is a CustomOpDef, save the existing abstract_fn so that
            # we can restore it after this contextmanager
            if opdef._abstract_fn is not None:
                old_fake_impls[op_str] = opdef._abstract_fn
            opdef.register_fake(fake_kernel)

        else:
            # Create a new library so that we can register a new fake impl.
            # These libraries will then be destroyed after the contextmanager,
            # which will automatically restore the previously registered fake
            # impls.
            newlib = torch.library.Library(namespace, "FRAGMENT")  # noqa: TOR901
            torch.library.register_fake(
                op_str, fake_kernel, lib=newlib, allow_override=True
            )
            libs.append(newlib)

    try:
        yield libs
    finally:
        # Destroying the libraries will automatically restore the previously
        # registered fake impls
        for lib in libs:
            lib._destroy()

        # Restore abstract_fns for CustomOpDefs
        for op_str, old_fake in old_fake_impls.items():
            opdef = _maybe_get_opdef(op_str)
            assert opdef is not None
            opdef.register_fake(old_fake)

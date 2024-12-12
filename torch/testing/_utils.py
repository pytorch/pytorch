# mypy: allow-untyped-defs
import contextlib

import torch


# Common testing utilities for use in public testing APIs.
# NB: these should all be importable without optional dependencies
# (like numpy and expecttest).


def wrapper_set_seed(op, *args, **kwargs):
    """Wrapper to set seed manually for some functions like dropout
    See: https://github.com/pytorch/pytorch/pull/62315#issuecomment-896143189 for more details.
    """
    with freeze_rng_state():
        torch.manual_seed(42)
        output = op(*args, **kwargs)

        if isinstance(output, torch.Tensor) and output.device.type == "lazy":
            # We need to call mark step inside freeze_rng_state so that numerics
            # match eager execution
            torch._lazy.mark_step()  # type: ignore[attr-defined]

        return output


@contextlib.contextmanager
def freeze_rng_state():
    # no_dispatch needed for test_composite_compliance
    # Some OpInfos use freeze_rng_state for rng determinism, but
    # test_composite_compliance overrides dispatch for all torch functions
    # which we need to disable to get and set rng state
    with torch.utils._mode_utils.no_dispatch(), torch._C._DisableFuncTorch():
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
    try:
        yield
    finally:
        # Modes are not happy with torch.cuda.set_rng_state
        # because it clones the state (which could produce a Tensor Subclass)
        # and then grabs the new tensor's data pointer in generator.set_state.
        #
        # In the long run torch.cuda.set_rng_state should probably be
        # an operator.
        #
        # NB: Mode disable is to avoid running cross-ref tests on thes seeding
        with torch.utils._mode_utils.no_dispatch(), torch._C._DisableFuncTorch():
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)  # type: ignore[possibly-undefined]
            torch.set_rng_state(rng_state)


# Used for tests as a function on module level
def _dummy_test_fn_with_module():
    pass

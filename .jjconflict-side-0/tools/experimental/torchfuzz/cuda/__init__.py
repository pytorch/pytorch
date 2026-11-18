"""CUDA device plugin for torchfuzz.

This is the reference device plugin loaded by ``torchfuzz.codegen`` when
``TORCHFUZZ_DEVICE_MODULE`` is unset (its default value is ``"torchfuzz.cuda"``).

Plugin contract
---------------

A device plugin is any importable Python module that exposes the following
two module-level functions::

    def register_codegen() -> dict[str, type[FuzzTemplate]]: ...
    def get_device_info() -> DeviceInfo: ...

where ``FuzzTemplate`` and ``DeviceInfo`` are imported from
``torchfuzz.codegen``.

* ``register_codegen()`` returns a mapping from template short-name (used by
  ``--template`` on the CLI) to the :class:`FuzzTemplate` subclass that
  implements that template.
* ``get_device_info()`` returns a :class:`DeviceInfo` instance carrying the
  device name (emitted into tensor descriptors) and an optional
  ``select_runtime_env(env)`` callback that mutates the subprocess environment
  ``runner.py`` uses when launching generated programs.

FuzzTemplate hooks
------------------

The core ``convert_graph_to_python_code`` is device-agnostic.  Plugins
customize emitted code by overriding the following hooks on
:class:`FuzzTemplate`:

================================  ========================================  ==================================================
Hook                              Default behaviour                         Where it is invoked
================================  ========================================  ==================================================
``imports_codegen``               ``[]``                                    Top of generated file
``flags_codegen``                 ``[]``                                    Top of generated file (after imports)
``args_codegen``                  Sentinel + per-arg ``torch.as_strided``   After the ``def fuzzed_program(...):`` body
``codegen_constant``              ``f"{output_name} = {expr}"``             ``operators/constant.py`` (TensorSpec branch)
``treat_constant_as_global``      ``False``                                 Decides if ``constant`` ops become function args
``return_codegen``                Multiplies by sentinel; ``.real`` for     Body's last lines (return statement)
                                  complex
``wrap_body``                     Returns lines unchanged                   After per-node code is emitted
``epilogue_codegen``              ``[]``                                    End of generated file
================================  ========================================  ==================================================

Per-template overrides for the CUDA plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`DefaultFuzzTemplate` - emits ``torch.set_default_device('cuda')``
  in ``flags_codegen``; otherwise inherits all base hooks.
* :class:`UnbackedFuzzTemplate` - same as default plus the
  ``capture_dynamic_output_shape_ops`` flag; tensor/scalar mix is 50/50.
* :class:`StreamFuzzTemplate` - overrides ``args_codegen`` to add
  ``requires_grad_(True)`` to float args, and overrides ``wrap_body`` to
  partition non-leaf operations across 2-3 ``torch.cuda.Stream()`` contexts
  with proper ``wait_stream`` / event-based synchronization.
* :class:`DTensorFuzzTemplate` - overrides ``imports_codegen`` /
  ``flags_codegen`` for distributed setup, ``args_codegen`` to wrap each arg
  via ``DTensor.from_local`` on a fake-PG 2D mesh, ``return_codegen`` to use
  ``.abs()`` (instead of ``.real``) for complex tensors,
  ``epilogue_codegen`` to destroy the process group, and
  ``codegen_constant`` to wrap constants via ``DTensor.from_local`` on cuda.
* :class:`DTensorFuzzPlacementsTemplate` - extends DTensor with randomized
  placements (Replicate/Shard/Partial); overrides ``treat_constant_as_global``
  to ``True`` so constants are materialized in ``args_codegen`` (via
  ``dist_tensor.full``), and ``codegen_constant`` becomes a comment marker.

Concrete implementations live in :mod:`torchfuzz.cuda._codegen`.

Runtime environment hook
------------------------

``select_runtime_env(env, *, exclude_primary_device=False)`` (registered as
``DeviceInfo.select_runtime_env``) is called by ``runner.py`` to customize the
environment used when launching the generated program in a subprocess.
``PYTHONPATH`` is already set by the runner before the hook is called; plugins
should not set it.  The CUDA implementation (:func:`_select_cuda_device`) picks
a random device from ``CUDA_VISIBLE_DEVICES`` (or ``torch.cuda.device_count()``
if unset) and narrows ``CUDA_VISIBLE_DEVICES`` to that single device.  When
*exclude_primary_device* is ``True`` and more than one GPU is available, GPU 0
is excluded to avoid contention with the orchestrating process.  Other plugins
should follow the same pattern (return a *new* dict, do not mutate the caller's
environment in place).
"""

from __future__ import annotations

import logging
import random

from torchfuzz.codegen import DeviceInfo, FuzzTemplate
from torchfuzz.cuda._codegen import (
    DefaultFuzzTemplate,
    DTensorFuzzPlacementsTemplate,
    DTensorFuzzTemplate,
    StreamFuzzTemplate,
    UnbackedFuzzTemplate,
)


logger: logging.Logger = logging.getLogger(__name__)


def register_codegen() -> dict[str, type[FuzzTemplate]]:
    """Return the CUDA plugin's template registry."""
    return {
        "default": DefaultFuzzTemplate,
        "dtensor": DTensorFuzzTemplate,
        "dtensor_placements": DTensorFuzzPlacementsTemplate,
        "unbacked": UnbackedFuzzTemplate,
        "streams": StreamFuzzTemplate,
    }


def _select_cuda_device(
    env: dict[str, str],
    *,
    exclude_primary_device: bool = False,
) -> dict[str, str]:
    """Pick a random CUDA device and narrow CUDA_VISIBLE_DEVICES to it.

    Returns a new env mapping (does not mutate ``env``).

    When *exclude_primary_device* is ``True`` and more than one GPU is
    available, GPU 0 is excluded from the candidate pool to avoid contention
    with the orchestrating process.
    """
    new_env = dict(env)
    cuda_visible_devices = new_env.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        devices = [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]
    else:
        # Default to all GPUs if not set
        try:
            import torch

            num_gpus = torch.cuda.device_count()
            device_range = (
                range(1, num_gpus)
                if exclude_primary_device and num_gpus > 1
                else range(num_gpus)
            )
            devices = [str(i) for i in device_range]
        except ImportError:
            devices = []

    if devices:
        selected_device = random.choice(devices)
        new_env["CUDA_VISIBLE_DEVICES"] = selected_device
        logger.info("Selected CUDA_VISIBLE_DEVICES=%s", selected_device)
    return new_env


def get_device_info() -> DeviceInfo:
    """Return the CUDA plugin's device metadata."""
    return DeviceInfo(device_name="cuda", select_runtime_env=_select_cuda_device)

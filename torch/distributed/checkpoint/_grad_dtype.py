import torch

from . import state_dict as _state_dict


def _init_optim_state(optim: torch.optim.Optimizer) -> None:
    for param_group in optim.param_groups:
        missing = [
            param
            for param in param_group["params"]
            if param.requires_grad and param not in optim.state
        ]
        if not missing:
            continue

        original_params = param_group["params"]
        original_lr = param_group.get("lr")
        param_group["params"] = missing
        try:
            for param in missing:
                grad_dtype = getattr(param, "grad_dtype", None) or param.dtype
                param.grad = torch.zeros_like(param, dtype=grad_dtype)
            if "lr" in param_group:
                param_group["lr"] = (
                    torch.zeros_like(original_lr)
                    if isinstance(original_lr, torch.Tensor)
                    else 0.0
                )
            optim.step(closure=None)
        finally:
            param_group["params"] = original_params
            if "lr" in param_group:
                param_group["lr"] = original_lr
            for param in missing:
                param.grad = None


_original_unflatten_optim_state_dict = _state_dict._unflatten_optim_state_dict


def _unflatten_optim_state_dict(optim, state_dict, info):
    params = [param for group in optim.param_groups for param in group["params"]]
    requires_grad = [param.requires_grad for param in params]
    temporarily_removed = {}
    flattened = "state" not in state_dict
    if flattened:
        missing = []
        for index, param in enumerate(params):
            saved_state = optim.state.get(param, {})
            if not saved_state or not requires_grad[index]:
                continue
            fqns = info.fqn_param_mapping.get(param, ())
            for fqn in fqns:
                for name in saved_state:
                    if f"state.{fqn}.{name}" not in state_dict:
                        missing.append((param, fqn))
                        break
        if missing and info.strict:
            raise RuntimeError(
                f"Missing optimizer state for parameter '{missing[0][1]}' in checkpoint. "
                "The parameter requires gradients but has no saved optimizer state. "
                "To load anyway, use StateDictOptions(strict=False)."
            )
        for param, _ in missing:
            if param not in temporarily_removed:
                temporarily_removed[param] = optim.state.pop(param)
    temporarily_enabled = [param for param in params if not param.requires_grad]
    try:
        for param in temporarily_enabled:
            param.requires_grad_(True)
        result = _original_unflatten_optim_state_dict(optim, state_dict, info)
        empty = [param for param, value in optim.state.items() if not value]
        for param in empty:
            optim.state.pop(param, None)
        if "state" in result:
            result["state"] = {
                key: value for key, value in result["state"].items() if value
            }
        return result
    finally:
        for param, value in temporarily_removed.items():
            optim.state[param] = value
        for param in temporarily_enabled:
            param.requires_grad_(False)


def _patch_consolidate_hf_safetensors() -> None:
    from . import _consolidate_hf_safetensors as consolidate

    def parse_input_metadata(input_files_data, output_files_data):
        from ._hf_utils import (
            DEFAULT_EXTRA_METADATA_KEY,
            DTYPE_KEY,
            SAVED_OFFSETS_KEY,
            SHAPE_KEY,
            _get_dcp_custom_metadata,
        )
        from safetensors.torch import _getdtype

        fqn_to_size_mapping = {}
        for file_data in input_files_data.values():
            metadata = file_data.metadata
            dcp_sharding_info = _get_dcp_custom_metadata(metadata)
            if not dcp_sharding_info:
                raise ValueError(
                    "No DCP custom metadata found in safetensors file. The file must be saved with DCP to be consolidated."
                )
            for key, val in metadata.items():
                if key == DEFAULT_EXTRA_METADATA_KEY:
                    continue
                sizes = val[SHAPE_KEY]
                offsets = dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                if key not in fqn_to_size_mapping:
                    fqn_to_size_mapping[key] = (
                        [size + offset for size, offset in zip(sizes, offsets)],
                        val[DTYPE_KEY],
                    )
                else:
                    cur_size = fqn_to_size_mapping[key][0]
                    for i in range(len(sizes)):
                        cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])
        for fqn, (tensor_size, dtype_str) in fqn_to_size_mapping.items():
            dtype = _getdtype(dtype_str)
            dtype_size = torch.empty((), dtype=dtype).element_size()
            for output_data in output_files_data.values():
                if fqn in output_data.fqn_data:
                    output_data.fqn_data[fqn] = consolidate._FqnData(
                        shape_in_file=tensor_size,
                        dtype_size=dtype_size,
                        dtype_str=dtype_str,
                    )

    consolidate._parse_input_metadata = parse_input_metadata


if not getattr(_state_dict, "_native_neo_grad_dtype_installed", False):
    _state_dict._unflatten_optim_state_dict = _unflatten_optim_state_dict
    _state_dict._native_neo_grad_dtype_installed = True

__all__ = ["_init_optim_state", "_patch_consolidate_hf_safetensors"]

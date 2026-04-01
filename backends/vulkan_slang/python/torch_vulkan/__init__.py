"""PyTorch Vulkan backend with Slang shaders for full training support."""

import importlib

_c_ext = None
_amp_state = {"enabled": False, "dtype": None}  # will set to torch.float16 in _register


def _ensure_loaded():
    global _c_ext
    if _c_ext is None:
        _c_ext = importlib.import_module("torch_vulkan._C")


def _register():
    """Entry point called by PyTorch to register the Vulkan backend."""
    _ensure_loaded()

    # Initialize AMP state dtype now that torch is available
    import torch
    _amp_state["dtype"] = torch.float16

    # Register torch.vulkan module so PyTorch can find it
    import sys
    import types
    vulkan_mod = types.ModuleType("torch.vulkan")
    vulkan_mod.is_available = is_available
    vulkan_mod.device_count = device_count
    vulkan_mod.get_device_name = get_device_name
    vulkan_mod.synchronize = synchronize
    vulkan_mod._is_in_bad_fork = lambda: False
    vulkan_mod.manual_seed_all = lambda seed: _c_ext._manual_seed(seed)

    # AMP support functions (required by torch.autocast)
    vulkan_mod.get_amp_supported_dtype = lambda: [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float32]
    vulkan_mod.is_autocast_enabled = lambda: _amp_state.get("enabled", False)
    vulkan_mod.set_autocast_enabled = lambda enabled: _amp_state.__setitem__("enabled", enabled)
    vulkan_mod.get_autocast_dtype = lambda: _amp_state.get("dtype", torch.float16)
    vulkan_mod.set_autocast_dtype = lambda dtype: _amp_state.__setitem__("dtype", dtype)

    sys.modules["torch.vulkan"] = vulkan_mod

    import torch
    torch.vulkan = vulkan_mod

    # Register serialization hooks for torch.save/load support
    _register_serialization()

    # Register Python-level overrides for ops with Tensor? args that fail
    # with None on PrivateUse1 dispatch (PyTorch dispatch key computation
    # calls .device() on undefined tensors)
    _register_optional_tensor_workarounds()

    # Generate convenience methods: tensor.vulkan(), model.vulkan()
    # Note: is_vulkan is a built-in PyTorch property (legacy Vulkan backend),
    # so we manually add the .vulkan() method instead of using generate_methods.
    import torch

    def _tensor_vulkan(self):
        return self.to(torch.device("vulkan", 0))

    def _module_vulkan(self, device=None):
        return self.to(torch.device("vulkan", device if device is not None else 0))

    if not hasattr(torch.Tensor, "vulkan"):
        torch.Tensor.vulkan = _tensor_vulkan
    if not hasattr(torch.nn.Module, "vulkan"):
        torch.nn.Module.vulkan = _module_vulkan

    # Fix is_vulkan to work with PrivateUse1
    @property
    def _is_vulkan(self):
        return self.device.type == "vulkan"

    torch.Tensor.is_vulkan = _is_vulkan

    # Register device module for AMP and other framework integration
    _register_device_module()

    # Register shutdown to run before Python module cleanup.
    # This ensures all Vulkan resources are released while VkDevice is valid.
    import atexit
    atexit.register(_c_ext._shutdown)


def _register_serialization():
    """Register hooks so torch.save/load works with Vulkan tensors."""
    import torch

    def _vulkan_tag(obj):
        """Tag function: identify Vulkan tensors for serialization."""
        if obj.device.type == "vulkan":
            return "vulkan"
        return None

    def _vulkan_deserialize(obj, location):
        """Deserialize: move tensor to Vulkan device if location is 'vulkan'."""
        if location and location.startswith("vulkan"):
            device_idx = 0
            if ":" in location:
                device_idx = int(location.split(":")[1])
            return obj.to(torch.device("vulkan", device_idx))
        return None

    try:
        torch.serialization.register_package(20, _vulkan_tag, _vulkan_deserialize)
    except (AttributeError, TypeError):
        # AttributeError: older PyTorch without register_package
        # TypeError: newer PyTorch where sort() fails on mixed function types
        pass


def _register_optional_tensor_workarounds():
    """Work around PyTorch dispatch bug: Tensor? args that are None cause
    'tensor does not have a device' error for PrivateUse1.
    We register Python overrides that convert None to zero tensors."""
    import torch
    import torch.nn.functional as F

    # Store originals so we can call through
    _orig_conv2d = F.conv2d
    _orig_layer_norm = F.layer_norm
    _orig_batch_norm = F.batch_norm
    _orig_group_norm = F.group_norm

    def _is_vulkan(t):
        """Check if tensor is on vulkan, safe during torch.compile tracing."""
        try:
            return t.device.type == "vulkan"
        except Exception:
            return False

    @torch.compiler.disable
    def _patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if _is_vulkan(input):
            if bias is None:
                bias = torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype)
            elif bias.dtype != input.dtype:
                bias = bias.to(dtype=input.dtype)
            if weight.dtype != input.dtype:
                weight = weight.to(dtype=input.dtype)
        return _orig_conv2d(input, weight, bias, stride, padding, dilation, groups)

    @torch.compiler.disable
    def _patched_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        if _is_vulkan(input):
            if weight is None:
                weight = torch.ones(normalized_shape, device=input.device, dtype=input.dtype)
            if bias is None:
                bias = torch.zeros(normalized_shape, device=input.device, dtype=input.dtype)
        return _orig_layer_norm(input, normalized_shape, weight, bias, eps)

    @torch.compiler.disable
    def _patched_batch_norm(input, running_mean, running_var, weight=None, bias=None,
                            training=False, momentum=0.1, eps=1e-5):
        if _is_vulkan(input):
            C = input.shape[1]
            if weight is None:
                weight = torch.ones(C, device=input.device, dtype=input.dtype)
            if bias is None:
                bias = torch.zeros(C, device=input.device, dtype=input.dtype)
            if running_mean is None:
                running_mean = torch.zeros(C, device=input.device, dtype=input.dtype)
            if running_var is None:
                running_var = torch.ones(C, device=input.device, dtype=input.dtype)
        return _orig_batch_norm(input, running_mean, running_var, weight, bias,
                                training, momentum, eps)

    @torch.compiler.disable
    def _patched_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
        if _is_vulkan(input):
            C = input.shape[1]
            if weight is None:
                weight = torch.ones(C, device=input.device, dtype=input.dtype)
            if bias is None:
                bias = torch.zeros(C, device=input.device, dtype=input.dtype)
        return _orig_group_norm(input, num_groups, weight, bias, eps)

    _orig_sdpa = F.scaled_dot_product_attention

    @torch.compiler.disable
    def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None, enable_gqa=False):
        if _is_vulkan(query):
            # Use the direct pybind11 binding which accepts attn_mask=None
            # without PyTorch dispatch key computation on the mask tensor.
            # This avoids "tensor does not have a device" and saves 1 dispatch
            # for causal SDPA (no zeros tensor creation needed).
            return _c_ext._sdpa(query, key, value,
                                 attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, scale=scale)
        return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, enable_gqa=enable_gqa)

    _orig_instance_norm = F.instance_norm

    @torch.compiler.disable
    def _patched_instance_norm(input, running_mean=None, running_var=None,
                                weight=None, bias=None, use_input_stats=True,
                                momentum=0.1, eps=1e-5):
        if _is_vulkan(input):
            # Instance norm = group norm with num_groups = num_channels
            C = input.shape[1]
            if weight is None:
                weight = torch.ones(C, device=input.device, dtype=input.dtype)
            if bias is None:
                bias = torch.zeros(C, device=input.device, dtype=input.dtype)
            return _orig_group_norm(input, C, weight, bias, eps)
        return _orig_instance_norm(input, running_mean, running_var, weight, bias,
                                    use_input_stats, momentum, eps)

    _orig_conv1d = F.conv1d

    @torch.compiler.disable
    def _patched_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if _is_vulkan(input):
            # Conv1d via conv2d: unsqueeze spatial dim, run conv2d, squeeze back
            input_4d = input.unsqueeze(2)  # [N, C, L] -> [N, C, 1, L]
            weight_4d = weight.unsqueeze(2)  # [C_out, C_in/g, K] -> [C_out, C_in/g, 1, K]
            s = (stride,) if isinstance(stride, int) else stride
            p = (padding,) if isinstance(padding, int) else padding
            d = (dilation,) if isinstance(dilation, int) else dilation
            if bias is None:
                bias = torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype)
            result = _orig_conv2d(input_4d, weight_4d, bias,
                                  stride=(1, s[0]), padding=(0, p[0]),
                                  dilation=(1, d[0]), groups=groups)
            return result.squeeze(2)
        return _orig_conv1d(input, weight, bias, stride, padding, dilation, groups)

    F.conv1d = _patched_conv1d
    F.conv2d = _patched_conv2d
    F.layer_norm = _patched_layer_norm
    F.batch_norm = _patched_batch_norm
    F.group_norm = _patched_group_norm
    F.scaled_dot_product_attention = _patched_sdpa
    F.instance_norm = _patched_instance_norm


def _register_device_module():
    """Register a device module with PyTorch for AMP and device APIs."""
    import torch

    class VulkanModule:
        @staticmethod
        def is_available():
            return _c_ext._is_available()

        @staticmethod
        def device_count():
            return _c_ext._device_count()

        @staticmethod
        def get_device_name(device_index=0):
            return _c_ext._get_device_name(device_index)

        @staticmethod
        def synchronize(device_index=0):
            _c_ext._synchronize(device_index)

        @staticmethod
        def _is_in_bad_fork():
            return False

        @staticmethod
        def manual_seed_all(seed):
            _c_ext._manual_seed(seed)

        # AMP support
        _autocast_enabled = False
        _autocast_dtype = torch.float16

        @staticmethod
        def get_amp_supported_dtype():
            return [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float32]

        @staticmethod
        def is_autocast_enabled():
            return VulkanModule._autocast_enabled

        @staticmethod
        def set_autocast_enabled(enabled):
            VulkanModule._autocast_enabled = enabled

        @staticmethod
        def get_autocast_dtype():
            return VulkanModule._autocast_dtype

        @staticmethod
        def set_autocast_dtype(dtype):
            VulkanModule._autocast_dtype = dtype

    try:
        torch._register_device_module("vulkan", VulkanModule)
    except (AttributeError, RuntimeError):
        pass


def is_available() -> bool:
    """Returns True if at least one Vulkan device is available."""
    _ensure_loaded()
    return _c_ext._is_available()


def device_count() -> int:
    """Returns the number of available Vulkan devices."""
    _ensure_loaded()
    return _c_ext._device_count()


def get_device_name(device_index: int = 0) -> str:
    """Returns the name of the specified Vulkan device."""
    _ensure_loaded()
    return _c_ext._get_device_name(device_index)


def synchronize(device_index: int = 0) -> None:
    """Waits for all pending Vulkan operations to complete."""
    _ensure_loaded()
    _c_ext._synchronize(device_index)


def empty_cache() -> None:
    """Release all cached Vulkan buffer allocations back to the system."""
    _ensure_loaded()
    _c_ext._empty_cache()


def memory_cached() -> int:
    """Returns the total bytes of cached (reusable) Vulkan buffer memory."""
    _ensure_loaded()
    return _c_ext._memory_cached()


def rope(input, theta: float = 10000.0):
    """Apply Rotary Position Embedding (RoPE) to input tensor [B, H, N, D]."""
    _ensure_loaded()
    return _c_ext.rope(input, theta)


def rms_norm(input, weight, eps: float = 1e-6):
    """Apply RMSNorm: weight * (input / sqrt(mean(input^2) + eps)).
    Used by Qwen3, Llama, etc. Supports autograd backward."""
    _ensure_loaded()
    return _c_ext.rms_norm(input, weight, eps)


def swiglu(gate, up):
    """Fused SwiGLU: silu(gate) * up. Single GPU dispatch instead of silu + mul.
    Used by Qwen3, Llama, Mistral MLP layers. Supports autograd backward."""
    _ensure_loaded()
    return _c_ext.swiglu(gate, up)



def scaled_bmm(q, k, scale: float):
    """Fused scaled BMM: scale * (q @ k.T) in a single GPU dispatch.
    Saves 1 dispatch vs separate bmm + mul_scalar.
    Useful for attention score computation: scaled_bmm(q, k, head_dim**-0.5).
    q: [B, M, K], k: [B, N, K] (NOT pre-transposed — we do k.T internally).
    Returns: [B, M, N] = scale * (q @ k.T). Supports autograd backward."""
    _ensure_loaded()
    return _c_ext.scaled_bmm(q, k, float(scale))


def add_rms_norm(residual, shortcut, weight, eps: float = 1e-6):
    """Fused Add + RMSNorm: h_new = residual + shortcut; normed = weight * (h_new / rms(h_new)).
    Returns (normed, h_new). Saves 1 dispatch vs separate add + rms_norm.
    Used in transformer residual connections: normed, h = add_rms_norm(h, layer_out, weight).
    Supports autograd backward through both outputs."""
    _ensure_loaded()
    return _c_ext.add_rms_norm(residual, shortcut, weight, float(eps))


def flash_attention(Q, K, V, scale: float, is_causal: bool = True, q_seq_major: bool = False):
    """Flash Attention: fused QK^T + softmax + @V in a single GPU dispatch.
    Eliminates the intermediate [B*H, N, S] attention weight matrix.
    Q: [B, H, N, D] (head-major) or [B, S, H, D] (seq-major, q_seq_major=True).
    K/V: [B, KV_H, S, D] (head-major) or [B, S, KV_H, D] (seq-major, q_seq_major=True).
    q_seq_major=True: pass tensors directly from linear+view without .transpose(1,2),
      saving 3 contiguous() copies per call (~3 dispatches per layer in a 4-layer model).
    Returns output [B, H, N, D]. Supports autograd backward (3 dispatches)."""
    _ensure_loaded()
    return _c_ext.flash_attention(Q, K, V, float(scale), bool(is_causal), bool(q_seq_major))


def rms_norm_gated(input, gate, weight, eps: float = 1e-6):
    """Fused RMSNormGated: weight * rms_norm(input) * silu(gate).
    Used by Qwen3.5-0.8B GatedDeltaNet layers (Qwen3_5RMSNormGated).
    Supports autograd backward."""
    _ensure_loaded()
    return _c_ext.rms_norm_gated(input, gate, weight, eps)


class SGD:
    """Fused SGD optimizer for Vulkan tensors.
    Single GPU dispatch per parameter instead of ~13 dispatches with torch.optim.SGD.
    API matches torch.optim.SGD."""

    def __init__(self, params, lr=0.01, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, foreach=None):
        import torch
        _ensure_loaded()
        if isinstance(params, torch.Tensor):
            params = [params]
        self.param_groups = [{"params": list(params), "lr": lr, "momentum": momentum,
                              "dampening": dampening, "weight_decay": weight_decay,
                              "nesterov": nesterov}]
        self.state = {}
        self.defaults = {"lr": lr, "momentum": momentum, "dampening": dampening,
                         "weight_decay": weight_decay, "nesterov": nesterov}

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, closure=None):
        import torch
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            if momentum == 0.0:
                # Batch mode: process up to 15 params per dispatch (no momentum).
                # sgd_batch15 uses 30 bindings + 184-byte push constants (NVIDIA supports 256).
                # C++ dispatch_shader falls back to sgd_batch (7 params) for n<=7 automatically.
                BATCH = 15
                batch_params = []
                batch_grads = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("VulkanSGD does not support sparse gradients")
                    # Only batch f32 params; fall back to per-param for other dtypes
                    if p.data.dtype == torch.float32:
                        batch_params.append(p.data)
                        batch_grads.append(p.grad)
                        if len(batch_params) == BATCH:
                            _c_ext._sgd_batch_step(batch_params, batch_grads, lr, weight_decay)
                            batch_params = []
                            batch_grads = []
                    else:
                        # Non-f32: use per-param fused step (handles cast internally)
                        buf = p.data  # dummy momentum buf
                        _c_ext._sgd_step(p.data, p.grad, buf, lr, 0.0, 0.0,
                                         weight_decay, False, False)
                if batch_params:
                    _c_ext._sgd_batch_step(batch_params, batch_grads, lr, weight_decay)
            else:
                # Momentum path: per-param fused step
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("VulkanSGD does not support sparse gradients")
                    state = self.state.setdefault(id(p), {})
                    has_buf = "momentum_buffer" in state
                    if not has_buf:
                        state["momentum_buffer"] = torch.zeros_like(p.data, dtype=torch.float32)
                    buf = state["momentum_buffer"]
                    _c_ext._sgd_step(p.data, grad, buf, lr, momentum, dampening,
                                     weight_decay, nesterov, True)

        return loss

    def add_param_group(self, param_group):
        self.param_groups.append(param_group)

    @property
    def param_groups_list(self):
        return self.param_groups


class AdamW:
    """Fused AdamW optimizer for Vulkan tensors.

    Uses decoupled weight decay. For bf16/f16 parameters, maintains float32
    master weights to avoid precision loss from quantization. Moment buffers
    are always float32.

    Args:
        params: iterable of Tensors on Vulkan device
        lr: learning rate (default: 1e-3)
        betas: (beta1, beta2) for moment estimates (default: (0.9, 0.999))
        eps: epsilon for numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay coefficient (default: 1e-2)
        master_weights: if True (default), maintain float32 master copies for
                        bf16/f16 parameters. Set False to skip (faster but
                        less numerically stable for half-precision training).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, master_weights=True):
        self.param_groups = [{"params": list(params), "lr": lr,
                               "betas": betas, "eps": eps, "weight_decay": weight_decay}]
        self.state = {}
        self._step = 0
        self._master_weights = master_weights

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, closure=None):
        import torch as _torch
        import math
        loss = None
        if closure is not None:
            loss = closure()

        self._step += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            bc1 = 1.0 - beta1 ** self._step
            bc2 = 1.0 - beta2 ** self._step

            # Separate f32 params (batchable) from half-precision (per-param due to master weight)
            batch_params, batch_grads, batch_m, batch_v = [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("VulkanAdamW does not support sparse gradients")
                state = self.state.setdefault(id(p), {})
                needs_master = (self._master_weights and
                                p.data.dtype in (_torch.float16, _torch.bfloat16))
                if "m" not in state:
                    state["m"] = _torch.zeros_like(p.data, dtype=_torch.float32)
                    state["v"] = _torch.zeros_like(p.data, dtype=_torch.float32)
                    if needs_master:
                        state["master"] = p.data.float()
                if needs_master:
                    master = state["master"]
                    grad_f32 = grad.float().to('vulkan') if grad.device.type != 'vulkan' else grad.float()
                    _c_ext._adamw_step(master, grad_f32, state["m"], state["v"],
                                       lr, beta1, beta2, eps, weight_decay, self._step)
                    p.data.copy_(master)
                else:
                    # Queue for batch dispatch
                    batch_params.append(p.data)
                    batch_grads.append(grad)
                    batch_m.append(state["m"])
                    batch_v.append(state["v"])
                    if len(batch_params) == 7:
                        _c_ext._adamw_batch_step(batch_params, batch_grads,
                                                  batch_m, batch_v,
                                                  lr, beta1, beta2, eps, weight_decay,
                                                  bc1, bc2)
                        batch_params, batch_grads, batch_m, batch_v = [], [], [], []
            if batch_params:
                _c_ext._adamw_batch_step(batch_params, batch_grads,
                                          batch_m, batch_v,
                                          lr, beta1, beta2, eps, weight_decay,
                                          bc1, bc2)

        return loss

    def add_param_group(self, param_group):
        self.param_groups.append(param_group)


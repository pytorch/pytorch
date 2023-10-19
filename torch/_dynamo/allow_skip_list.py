import functools

torch_ctx_manager_classes = {
    "torch.ExcludeDispatchKeyGuard",
    "torch._C.DisableTorchFunction",
    "torch._C.DisableTorchFunctionSubclass",
    "torch._C._AutoDispatchBelowAutograd",
    "torch._C._DisableAutocast",
    "torch._C._DisableFuncTorch",
    "torch._C._DisablePythonDispatcher",
    "torch._C._DisableTorchDispatch",
    "torch._C._EnablePreDispatch",
    "torch._C._EnablePythonDispatcher",
    "torch._C._EnableTorchFunction",
    "torch._C._ExcludeDispatchKeyGuard",
    "torch._C._ForceDispatchKeyGuard",
    "torch._C._IncludeDispatchKeyGuard",
    "torch._C._InferenceMode",
    "torch._C._RestorePythonTLSSnapshot",
    "torch._C._SetExcludeDispatchKeyGuard",
    "torch._C._profiler._RecordFunctionFast",
    "torch._decomp.decompositions_for_rng.PhiloxStateTracker",
    "torch._subclasses.fake_tensor.FakeTensorMode",
    "torch._subclasses.functional_tensor.FunctionalTensorMode",
    "torch.amp.autocast_mode.autocast",
    "torch.ao.nn.sparse.quantized.utils.LinearBlockSparsePattern",
    "torch.autograd.anomaly_mode.detect_anomaly",
    "torch.autograd.anomaly_mode.set_detect_anomaly",
    "torch.autograd.forward_ad._set_fwd_grad_enabled",
    "torch.autograd.forward_ad.dual_level",
    "torch.autograd.grad_mode._force_original_view_tracking",
    "torch.autograd.grad_mode._unsafe_preserve_version_counter",
    "torch.autograd.grad_mode.enable_grad",
    "torch.autograd.grad_mode.inference_mode",
    "torch.autograd.grad_mode.no_grad",
    "torch.autograd.grad_mode.set_grad_enabled",
    "torch.autograd.grad_mode.set_multithreading_enabled",
    "torch.autograd.graph.saved_tensors_hooks",
    "torch.autograd.profiler.emit_itt",
    "torch.autograd.profiler.emit_nvtx",
    "torch.autograd.profiler.profile",
    "torch.autograd.profiler.record_function",
    "torch.autograd.profiler_legacy.profile",
    "torch.backends.mkl.verbose",
    "torch.backends.mkldnn.verbose",
    "torch.cpu.StreamContext",
    "torch.cpu.amp.autocast_mode.autocast",
    "torch.cuda.StreamContext",
    "torch.cuda._DeviceGuard",
    "torch.cuda.amp.autocast_mode.autocast",
    "torch.cuda.device",
    "torch.cuda.graphs.graph",
    "torch.device",
    "torch.distributed.autograd.context",
    "torch.distributed.rpc.server_process_global_profiler._server_process_global_profile",
    "torch.hub._Faketqdm",
    "torch.jit._ir_utils._InsertPoint",
    "torch.jit._script.RecursiveScriptClass",
    "torch.jit.strict_fusion",
    "torch.onnx._internal.diagnostics.infra.context.DiagnosticContext",
    "torch.onnx._internal.fx.patcher.ONNXTorchPatcher",
    "torch.overrides.TorchFunctionMode",
    "torch.package.package_exporter.PackageExporter",
    "torch.profiler.profiler.profile",
    "torch.serialization._opener",
    "torch.sparse.check_sparse_tensor_invariants",
    "torch.utils._contextlib._DecoratorContextManager",
    "torch.utils._device.DeviceContext",
    "torch.utils._python_dispatch.TorchDispatchMode",
    "torch.utils.data.datapipes._decorator.guaranteed_datapipes_determinism",
    "torch.utils.data.datapipes._decorator.runtime_validation_disabled",
    "torch.utils.data.datapipes.dataframe.dataframes.CaptureLikeMock",
    "torch.utils.hooks.RemovableHandle",
}


@functools.lru_cache(None)
def get_torch_ctx_manager_classes():
    classes = set()
    for cls_name in torch_ctx_manager_classes:
        module_name, class_name = cls_name.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        class_obj = getattr(module, class_name)
        classes.add(class_obj)
    return classes


def is_torch_ctx_manager_class(class_obj):
    return class_obj in get_torch_ctx_manager_classes()

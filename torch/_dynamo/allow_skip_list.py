import functools

torch_ctx_manager_classes = {
    "torch.device",
    "torch.ExcludeDispatchKeyGuard",
    "torch.utils.hooks.RemovableHandle",
    "torch.utils._contextlib._DecoratorContextManager",
    "torch.utils._python_dispatch.TorchDispatchMode",
    "torch.utils.data.datapipes._decorator.guaranteed_datapipes_determinism",
    "torch.utils.data.datapipes._decorator.runtime_validation_disabled",
    "torch.utils.data.datapipes.dataframe.dataframes.CaptureLikeMock",
    "torch.distributed.autograd.context",
    "torch.autograd.profiler_legacy.profile",
    "torch.distributed.rpc.server_process_global_profiler._server_process_global_profile",
    "torch.overrides.TorchFunctionMode",
    "torch.utils._device.DeviceContext",
    "torch.serialization._opener",
    "torch.amp.autocast_mode.autocast",
    "torch.cuda.graphs.graph",
    "torch.cuda._DeviceGuard",
    "torch.cuda.device",
    "torch.cuda.StreamContext",
    "torch.cuda.amp.autocast_mode.autocast",
    "torch.sparse.check_sparse_tensor_invariants",
    "torch.autograd.forward_ad._set_fwd_grad_enabled",
    "torch.autograd.forward_ad.dual_level",
    "torch.autograd.grad_mode.no_grad",
    "torch.autograd.grad_mode.enable_grad",
    "torch.autograd.grad_mode.set_grad_enabled",
    "torch.autograd.grad_mode.inference_mode",
    "torch.autograd.grad_mode.set_multithreading_enabled",
    "torch.autograd.grad_mode._force_original_view_tracking",
    "torch.autograd.grad_mode._unsafe_preserve_version_counter",
    "torch.autograd.graph.saved_tensors_hooks",
    "torch.autograd.anomaly_mode.detect_anomaly",
    "torch.autograd.anomaly_mode.set_detect_anomaly",
    "torch.autograd.profiler.profile",
    "torch.autograd.profiler.record_function",
    "torch.autograd.profiler.emit_itt",
    "torch.autograd.profiler.emit_nvtx",
    "torch.package.package_exporter.PackageExporter",
    "torch.cpu.amp.autocast_mode.autocast",
    "torch.cpu.StreamContext",
    "torch.backends.mkl.verbose",
    "torch.backends.mkldnn.verbose",
    "torch.jit._script.RecursiveScriptClass",
    "torch.jit._ir_utils._InsertPoint",
    "torch.jit.strict_fusion",
    "torch.hub._Faketqdm",
    "torch.profiler.profiler.profile",
    "torch.ao.nn.sparse.quantized.utils.LinearBlockSparsePattern",
    "torch._decomp.decompositions_for_rng.PhiloxStateTracker",
    "torch._subclasses.fake_tensor.FakeTensorMode",
    "torch._subclasses.functional_tensor.FunctionalTensorMode",
    "torch.onnx._internal.diagnostics.infra.context.DiagnosticContext",
    "torch.onnx._internal.fx.patcher.ONNXTorchPatcher",
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

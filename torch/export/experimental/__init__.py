import copy
import dataclasses
import functools
import os
import types
import typing
import typing_extensions
import zipfile
from pathlib import Path

import torch
from torch.export.experimental._utils import _get_main_cpp_file, _get_make_file
from torch.export.exported_program import _decompose_exported_program


_InputT = typing_extensions.ParamSpec("_InputT")
_RetT = typing.TypeVar("_RetT")


__all__ = []  # type: ignore[var-annotated]


def _copy_graph_module_and_signature(
    ep: torch.export.ExportedProgram,
) -> tuple[torch.fx.GraphModule, torch.export.graph_signature.ExportGraphSignature]:
    # copy.deepcopy lets the objects override __deepcopy__ methods with graph_copy() and node_copy(),
    # and this can break placeholder names in some particular cases.
    # For example, node copying will avoid Python keywords like 'input', suffixing and renaming to 'input_1'.
    # So we manually overwrite placeholder names by reading the old graph.
    gm = copy.deepcopy(ep.graph_module)
    new_graph_signature = copy.deepcopy(ep.graph_signature)

    # iterate over old/new graph modules
    for old_gm, new_gm in zip(ep.graph_module.modules(), gm.modules()):  # type: ignore[union-attr]
        old_phs = [node for node in old_gm.graph.nodes if node.op == "placeholder"]
        new_phs = [node for node in new_gm.graph.nodes if node.op == "placeholder"]
        # iterate over placeholders
        if len(old_phs) != len(new_phs):
            raise AssertionError(
                f"Number of old placeholders ({len(old_phs)}) does not match "
                f"new placeholders ({len(new_phs)})"
            )
        for old_node, new_node in zip(old_phs, new_phs):
            new_node.name = old_node.name

    return gm, new_graph_signature


def _remove_detach_pass(
    gm: torch.fx.GraphModule, sig: torch.export.graph_signature.ExportGraphSignature
) -> None:
    with gm._set_replace_hook(sig.get_replace_hook()):
        for node in list(reversed(gm.graph.nodes)):
            if node.op != "call_function":
                continue
            if (
                node.target is torch.ops.aten.detach.default
                and len(node.users) == 1
                and next(iter(node.users)).target is torch.ops.aten.detach.default
            ):
                next(iter(node.users)).replace_all_uses_with(node)

    gm.graph.eliminate_dead_code()
    gm.recompile()


def _export_forward_backward(
    ep: torch.export.ExportedProgram, joint_loss_index: int = 0
) -> torch.export.ExportedProgram:
    """
    WARNING: This API is highly unstable and will be subject to change in the future.
    """
    from torch._decomp import core_aten_decompositions

    ep = _decompose_exported_program(
        ep,
        cia_to_decomp={},
        python_decomp_table=core_aten_decompositions(),
        joint_loss_index=joint_loss_index,
        # For serialization purpose, we don't want to decompose custom triton ops.
        # If users would like to decompose custom triton ops, they could do it
        # with run_decompositions() API.
        decompose_custom_triton_ops=False,
    )
    gm, new_graph_signature = _copy_graph_module_and_signature(ep)
    _remove_detach_pass(gm, new_graph_signature)

    return ep._update(gm, new_graph_signature)


def _sticky_export(
    forward_func: typing.Callable[_InputT, _RetT],
    dynamic_shapes_callback: typing.Callable[
        _InputT, list[typing.Any] | dict[str, typing.Any] | tuple[typing.Any, ...]
    ]
    | None = None,
) -> typing.Callable[_InputT, _RetT]:
    """
    Lazily export the model on first forward call.
    Usage:
        model.forward = _sticky_export(model.forward, dynamic_shapes_callback=callback)
    """
    model = forward_func.__self__  # type: ignore[attr-defined]
    original_forward = forward_func.__func__  # type: ignore[attr-defined]

    @functools.wraps(forward_func)
    def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        # Unpatch forward to avoid recursion during export
        model.forward = types.MethodType(original_forward, model)

        dynamic_shapes_spec = None
        if dynamic_shapes_callback:
            dynamic_shapes_spec = dynamic_shapes_callback(*args, **kwargs)

        try:
            exported = torch.export.export(
                model,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes_spec,
            ).module()
            wrapper._exported_artifact = exported  # type: ignore[attr-defined]
        finally:
            # Restore the wrapper after export
            model.forward = wrapper

        return exported(*args, **kwargs)

    return wrapper


@dataclasses.dataclass
class _ExportMethod:
    overloads: dict[str, torch.export.ExportedProgram]
    fallbacks: list[torch.export.ExportedProgram]


class _ExportPackage:
    """
    An export package is a collection of torch.export()-ed PyTorch models consisting of
    a list of exported methods and their corresponding overloads. ExportPackage is introduced
    on top of torch.export() to support the following use cases:
        - Exporting a model with multiple methods if a model has multiple independent parts.
        - Exporting a function with multiple overloads based on tensor shapes or other metadata.

    ExportPackage is designed to contain multiple methods (associated with method names) and for
    each method, it can have multiple overloads (associated with overload names).

    Here is an example of the data structure for an ExportPackage:
    ```
    ExportPackage(
        methods={
            "decoder": ExportMethod(
                overloads={
                    "prefill": ExportedProgram(...),
                    "decode": ExportedProgram(...),
                },
                fallbacks=[],
            ),
            "encoder": ExportMethod(overloads={}, fallbacks=[ExportedProgram(...)]),
        },
    )
    ```

    To export a model into an ExportPackage, users can use the exporter API provided by ExportPackage.
    Exporter is a decorator that takes a callable and returns a wrapper. The wrapper will export the
    function into an ExportPackage, when it's invoked with some sample inputs (similar to how
    torch.compile() works). For more details, please refer to the document on .exporter() method.

    This design allows users to decouple the exported callables from the actual sample inputs which can
    be helpful for use cases where the exported callable is hidden behind helper functions or when sample
    inpusts are hard to get.

    NOTE: This is an experimental API and anything can be changed in the future.

    Example usage:
    ```
        def fn(x):
            return x + 1

        def main(f, x):
            x += 1
            ret = f(x)
            return ret + 1

        package = ExportPackage()
        main(package.exporter(fn), torch.randn(3, 2))
    ```

    """

    def __init__(self) -> None:
        self.methods: dict[str, _ExportMethod] = {}

    def _exporter(
        self,
        method: str,
        fn: typing.Callable[_InputT, _RetT],
        *,
        fallback: str = "once",
    ) -> typing.Callable[_InputT, _RetT]:
        """
        A function/module decorator that sets up a callable to be exported later invoked.
        By default the exporter will only trigger torch.export for once and error on
        later invocations. To customize this behavior, users have the following two options:
          1. Call .define_overload() method on the returned wrapper to define an overload.
          2. Adjust the fallback policy using `fallback` argument.

        An "overload" is a named branch for an ExportMethod with a user defined precondition,
        typically based on input tensor shapes. It's up to a downstream backend implementation
        of ExportMethod to respect the precondition later in inference.

        define_overload() takes arguments like the following:
          - A name, for indexing purposes in a backend.
          - A callable (spec) that:
            - Has the same model input signature as the original model code.
            - Returns an optional dynamic shape spec.

        Exporter will only export an overload when the spec callable successfully returns
        a result without raising AssertionError.

        For example:
        ```
        package = ExportPackage()


        def prefill(x, xa, kv_cache):
            assert x.shape[1] == 3
            assert kv_cache == {}


        def decode(x, xa, kv_cache):
            assert x.shape[1] > 1
            assert len(kv_cache) > 0
            return {...}  # dynamic shape specs here


        exporter = (
            package.exporter(decoder)
            .define_overload("prefill", prefill)
            .define_overload("decode", decode)
        )
        ```

        A "fallback" is exported when no overload precondition matches a given set of sample
        inputs. Overloads should
        Fallbacks don't have names and are ordered in a list. It's up to a backend to decide
        which fallback is used amony multiple ones.

        A reference backend implementation of ExportMethod may look like the following:
        ```
        def execute(method: ExportMethod, *args, **kwargs):
            for overload in method.overloads:
                if match_precondition(overload, *args, **kwargs):
                    return execute_overload(overload, *args, **kwargs)
            for fallback in method.fallbacks:
                if match_precondition(fallback, *args, **kwargs):
                    return execute_fallback(fallback, *args, **kwargs)
        ```

        Args:
            method(str): The method name for an exported part of PyTorch model. This
                         will be saved together with the exported/compiled artifacts
                         in any serialization format and can be used as the key to
                         index ExportPackage methods later.
            fn(callable): A PyTorch function/module to be exported.
            fallback(str): The fallback policy to decide when to call torch.export
              - "once" is the default policy. Under this policy a PyTorch program is assumed
                to be only called once later and an error will be raised for subsequent
                runs.
              - "error" means the ExportMethod will never have any fallbacks, meaning
                users should define all the possible overloads ahead of time.

        """

        fallbacks: list[torch.export.ExportedProgram] = []
        specs: dict[str, typing.Callable[_InputT, typing.Any]] = {}
        overloads: dict[str, torch.export.ExportedProgram] = {}
        self.methods[method] = _ExportMethod(fallbacks=fallbacks, overloads=overloads)

        @functools.wraps(fn)
        def _exporter_context(*args, **kwargs):  # type: ignore[no-untyped-def]
            import torch.export._wrapper_utils

            model: torch.nn.Module
            if not isinstance(fn, torch.nn.Module):
                model = torch.export._wrapper_utils._WrapperModule(fn)
            else:
                model = fn

            for k, v in specs.items():
                try:
                    if isinstance(fn, torch.nn.Module):
                        dynamic_shapes = v(fn, *args, **kwargs)  # type: ignore[arg-type]
                    else:
                        # pyrefly: ignore [invalid-param-spec]
                        dynamic_shapes = v(*args, **kwargs)
                except AssertionError:
                    continue
                if k not in overloads:
                    ep = torch.export.export(
                        model, args, kwargs, dynamic_shapes=dynamic_shapes
                    )
                    overloads[k] = ep
                ep = overloads[k]
                return ep.module()(*args, **kwargs)

            if fallback == "error":
                raise RuntimeError(
                    f"Exporter: Cannot export fallback {fn} when fallback policy is set to 'error',"
                    + "please specify an overload or adjust the fallback policy."
                )
            elif fallback == "once":
                if len(fallbacks) > 0:
                    raise RuntimeError(
                        f"Exporter: Cannot export {fn} more than once, "
                        + "please specify an overload or adjust the fallback policy."
                    )
            else:
                raise RuntimeError(f"Unknown fallback policy: {fallback}")
            ep = torch.export.export(model, args, kwargs)

            fallbacks.append(ep)
            return ep.module()(*args, **kwargs)

        if isinstance(fn, torch.nn.Module):
            _exporter_context = torch._dynamo.eval_frame.OptimizedModule(  # type: ignore[assignment] # noqa: F811
                fn,
                lambda _: _exporter_context,  # type: ignore[arg-type]
            )

        def _define_overload(
            overload: str, spec: typing.Callable[_InputT, typing.Any]
        ) -> typing.Any:
            if overload in specs:
                raise AssertionError(f"Overload '{overload}' already exists in specs")
            if not callable(spec):
                raise AssertionError(f"spec must be callable, but got {type(spec)}")
            if not overload.isidentifier():
                raise AssertionError(
                    f"Overload '{overload}' is not a valid Python identifier"
                )
            specs[overload] = spec
            return _exporter_context

        if hasattr(fn, "_define_overload"):
            raise AssertionError("fn already has a '_define_overload' attribute")
        _exporter_context._define_overload = _define_overload  # type: ignore[attr-defined]

        # pyrefly: ignore [bad-return]
        return _exporter_context

    @property
    def _method_overloads(
        self,
    ) -> typing.Iterator[tuple[str, torch.export.ExportedProgram]]:
        for method, method_data in self.methods.items():
            for overload, ep in method_data.overloads.items():
                yield f"{method}:{overload}", ep

    def _compiled_and_package(
        self,
        f: torch.types.FileLike,
        standalone: bool = False,
        package_example_inputs: bool = False,
    ) -> None:
        options: dict[str, typing.Any] = {
            "aot_inductor.package": True,
            "aot_inductor.package_cpp_only": True,
            "always_keep_tensor_constants": True,
            # we'll change this back to False once we enable weight deduping for standalone mode
            "aot_inductor.package_constants_in_so": standalone,
            "aot_inductor_mode.compile_standalone": standalone,
        }
        aoti_files_map = {}
        model_names = []
        for name, ep in self._method_overloads:
            name = name.replace(":", "__")
            model_names.append(name)
            options["aot_inductor.model_name_for_generated_files"] = name
            aoti_files = torch._inductor.aot_compile(
                ep.module(),  # type: ignore[arg-type]
                ep.example_inputs[0],
                kwargs=ep.example_inputs[1],
                options=options,
            )
            # pyrefly: ignore [unsupported-operation]
            aoti_files_map[name] = aoti_files

        from torch._inductor.package import package

        pt2_path = package.package_aoti(
            f,
            aoti_files_map,  # type: ignore[arg-type]
        )

        if not standalone:
            return

        if not isinstance(pt2_path, str):
            raise AssertionError(
                f"Expected pt2_path to be a string, but got {type(pt2_path)}"
            )
        base_directory = os.path.dirname(pt2_path)
        package_name = os.path.basename(pt2_path)[:-4]
        with (
            zipfile.ZipFile(pt2_path, "r") as zip_ref,
        ):
            zip_ref.extractall(base_directory)

        example_inputs_map: dict[str, int] | None = (
            {} if package_example_inputs else None
        )
        use_cuda = False
        for name, ep in self._method_overloads:
            name = name.replace(":", "__")
            # TODO: also dump kwargs
            # TODO: currently only support list of Tensors and they need to be on the same device
            if not ep.example_inputs:
                continue
            for inp in ep.example_inputs[0]:
                if isinstance(inp, torch.Tensor) and inp.device.type == "cuda":
                    # TODO: more carefully determine the device type
                    use_cuda = True
            if package_example_inputs:
                if example_inputs_map is None:
                    raise AssertionError(
                        "example_inputs_map cannot be None when package_example_inputs is True"
                    )
                example_inputs_map[name] = len(ep.example_inputs[0])
                for i, t in enumerate(ep.example_inputs[0]):
                    path = Path(base_directory) / f"{name}_input_{i}.pt"
                    torch.save(t, path)

        # Detect if ROCm is being used
        is_hip = torch.version.hip is not None
        cmake_file_str = _get_make_file(package_name, model_names, use_cuda, is_hip)

        with open(Path(base_directory) / "CMakeLists.txt", "w") as file:
            file.write(cmake_file_str)

        main_file_str = _get_main_cpp_file(
            package_name, model_names, use_cuda, example_inputs_map, is_hip
        )
        with open(Path(base_directory) / "main.cpp", "w") as file:
            file.write(main_file_str)

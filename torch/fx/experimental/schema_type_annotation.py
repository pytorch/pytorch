# mypy: allow-untyped-defs
import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.fx
from torch._jit_internal import boolean_dispatched
from torch.fx import Transformer
from torch.fx.node import Argument, Target
from torch.fx.operator_schemas import _torchscript_type_to_python_type


class AnnotateTypesWithSchema(Transformer):
    """
    Use Python function signatures to annotate types for `Nodes` within an FX graph.
    This pulls out Python function signatures for:

        1. Standard `torch.nn` Module calls
        2. `torch.nn.functional` calls
        3. Attribute fetches via `get_attr`

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = AnnotateTypesWithSchema(traced).transform()

    """

    def __init__(
        self,
        module: torch.nn.Module,
        annotate_functionals: bool = True,
        annotate_modules: bool = True,
        annotate_get_attrs: bool = True,
    ):
        super().__init__(module)
        self.annotate_functionals = annotate_functionals
        self.annotate_modules = annotate_modules
        self.annotate_get_attrs = annotate_get_attrs

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ):
        python_ret_type = None
        if self.annotate_functionals and target.__module__ == "torch.nn.functional":
            target_for_analysis = target
            if target in boolean_dispatched:
                # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
                # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
                # branches of the dispatch have exactly the same signature. If they do, use the `true`
                # branch signature for analysis. Otherwise, leave this un-normalized
                assert not isinstance(target, str)
                dispatched = boolean_dispatched[target]
                if_true, if_false = dispatched["if_true"], dispatched["if_false"]
                # TODO: can we emit the union of these? What are the implications on TorchScript
                # compilation?
                if (
                    inspect.signature(if_true).return_annotation
                    != inspect.signature(if_false).return_annotation
                ):
                    return super().call_function(target, args, kwargs)
                target_for_analysis = if_true

            python_ret_type = self._extract_python_return_type(target_for_analysis)

        return_proxy = super().call_function(target, args, kwargs)
        return_proxy.node.type = (
            return_proxy.node.type if return_proxy.node.type else python_ret_type
        )
        return return_proxy

    def call_module(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ):
        python_ret_type = None
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        if self.annotate_modules and hasattr(submod.__class__, "__name__"):
            classname = submod.__class__.__name__
            if getattr(torch.nn, classname, None) == submod.__class__:
                python_ret_type = self._extract_python_return_type(submod.forward)
        return_proxy = super().call_module(target, args, kwargs)
        return_proxy.node.type = (
            return_proxy.node.type if return_proxy.node.type else python_ret_type
        )
        return return_proxy

    def get_attr(
        self,
        target: torch.fx.node.Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Any],
    ):
        attr_proxy = super().get_attr(target, args, kwargs)

        if self.annotate_get_attrs:
            module_itr = self.module
            assert isinstance(target, str)
            atoms = target.split(".")
            for i, atom in enumerate(atoms):
                if not hasattr(module_itr, atom):
                    raise RuntimeError(
                        f'Node referenced nonextent target {".".join(atoms[:i])}!'
                    )
                module_itr = getattr(module_itr, atom)

            maybe_inferred_ts_type = torch._C._jit_try_infer_type(module_itr)
            if maybe_inferred_ts_type.success():
                python_type = _torchscript_type_to_python_type(
                    maybe_inferred_ts_type.type()
                )
                attr_proxy.node.type = (
                    python_type if not attr_proxy.node.type else attr_proxy.node.type
                )

        return attr_proxy

    def _extract_python_return_type(self, target: Target) -> Optional[Any]:
        """
        Given a Python call target, try to extract the Python return annotation
        if it is available, otherwise return None

        Args:

            target (Callable): Python callable to get return annotation for

        Returns:

            Optional[Any]: Return annotation from the `target`, or None if it was
                not available.
        """
        assert callable(target)
        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            return None

        return (
            sig.return_annotation
            if sig.return_annotation is not inspect.Signature.empty
            else None
        )

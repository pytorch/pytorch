# mypy: allow-untyped-defs
import builtins
import functools
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.fx


def embedding_override(self, input):
    return torch.empty(*input.shape, self.weight.shape[-1], device="meta")


def nn_layernorm_override(self, input):
    return input


def torch_relu_override(x):
    return x


def torch_nn_relu_override(self, x):
    return x


def functional_relu_override(x, inplace=False):
    assert not inplace, "dont support inplace functional.relu for metatensor analysis"
    return x


def torch_where_override(condition, x, y):
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


def torch_abs_override(input, *, out=None):
    assert out is None, "Dont support in-place abs for MetaTensor analysis"
    return input


manual_meta_overrides: dict[Callable, Callable] = {
    torch.nn.Embedding: embedding_override,
    torch.nn.LayerNorm: nn_layernorm_override,
    torch.relu: torch_relu_override,
    torch.nn.functional.relu: functional_relu_override,
    torch.nn.ReLU: torch_nn_relu_override,
    torch.where: torch_where_override,
    torch.abs: torch_abs_override,
}


def gen_constructor_wrapper(target):
    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


class MetaProxy(torch.fx.Proxy):
    def install_tensor_meta(self, tensor_meta):
        self._tensor_meta = tensor_meta

    def size(self, dim=None):
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.size(*[dim] if dim else [])
        return self.tracer.create_proxy(
            "call_method", "size", (self, dim) if dim else (self,), {}
        )

    def dim(self):
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.dim()
        return self.tracer.create_proxy("call_method", "dim", (self,), {})

    @property
    def shape(self):
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.shape
        return self.tracer.create_proxy(
            "call_function", builtins.getattr, (self, "shape"), {}
        )

    @property
    def dtype(self):
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.dtype
        return self.tracer.create_proxy(
            "call_function", builtins.getattr, (self, "dtype"), {}
        )

    @property
    def device(self):
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    def __getattr__(self, k):
        if k == "_tensor_meta":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return MetaAttribute(self, k)


class MetaAttribute(MetaProxy):
    def __init__(self, root, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

    @property
    def node(self):  # type: ignore[override]
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs
        )


class MetaDeviceAttribute(MetaAttribute):
    pass


def proxys_to_metas(v):
    if isinstance(v, MetaDeviceAttribute):
        return "meta"
    if isinstance(v, torch.fx.Proxy):
        assert isinstance(v, MetaProxy), f"Expected MetaProxy but got {type(v)}"
        assert hasattr(v, "_tensor_meta"), "MetaProxy does not have an associated meta"
        return v._tensor_meta
    return v


class MetaTracer(torch.fx.Tracer):
    allow_insert_stateless_mods: bool = True

    _TORCH_METHODS_TO_PATCH = ["arange", "zeros", "ones", "full_like", "eye"]

    def create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        rv = super().create_proxy(
            kind,
            target,
            args,
            kwargs,
            name,
            type_expr,
            # pyrefly: ignore  # bad-argument-type
            proxy_factory_fn,
        )

        if kind == "placeholder" and target in self.meta_args:
            rv.install_tensor_meta(self.meta_args[target])
            return rv

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = torch.fx.node.map_aggregate(args, proxys_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, proxys_to_metas)

            if kind == "call_function":
                meta_target = manual_meta_overrides.get(target, target)
                # pyrefly: ignore  # not-callable
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_method":
                meta_target = getattr(args_metas[0], target)  # type: ignore[index]
                meta_out = meta_target(*args_metas[1:], **kwargs_metas)  # type: ignore[index]
            elif kind == "call_module":
                assert hasattr(self, "orig_forward")
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in manual_meta_overrides:
                        meta_out = manual_meta_overrides[mod_type](
                            mod, *args_metas, **kwargs_metas
                        )  # type: ignore[misc, arg-type]
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    assert isinstance(attr_itr, torch.Tensor)
                    meta_out = attr_itr.to(device="meta")
                finally:
                    self._disable_module_getattr = False
            else:
                return rv

            # TODO
            assert isinstance(rv, torch.fx.Proxy), "Dont support composite output yet"
            rv.install_tensor_meta(meta_out)
        except Exception as e:
            warnings.warn(f"Could not compute metadata for {kind} target {target}: {e}")

        return rv

    def getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:
            return super().getattr(attr, attr_val, parameter_proxy_cache)

    def call_module(self, m, forward, args, kwargs):
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    def _insert_module_as_submodule(self, mod: torch.nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        while hasattr(self.root, path):
            path = f"{mod_name}_{idx}"
            idx += 1

        self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: torch.nn.Module) -> str:
        try:
            return super().path_of_module(mod)
        except NameError:
            if (
                self.allow_insert_stateless_mods
                and len(list(mod.parameters())) == 0
                and len(list(mod.buffers())) == 0
            ):
                path = self._insert_module_as_submodule(mod)
                self.prev_module = path
                return path
            raise

    def proxy(self, node):
        return MetaProxy(node, self)

    def trace(self, root, meta_args: dict[str, torch.Tensor], concrete_args=None):  # type: ignore[override]
        assert isinstance(meta_args, dict)
        self.meta_args = meta_args

        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target))
            for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            graph = super().trace(root, concrete_args)
            graph._tracer_extras = {"meta_args": meta_args}
            return graph
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)


def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    meta_args: Optional[dict[str, torch.Tensor]] = None,
    concrete_args: Optional[dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    tracer = MetaTracer()
    graph = tracer.trace(root, meta_args, concrete_args)  # type: ignore[arg-type]
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    gm = torch.fx.GraphModule(tracer.root, graph, name)
    return gm

import torch
import torch.fx
import warnings
import functools

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
            return proxy.tracer.create_proxy('call_function', target, args, kwargs)
        else:
            return target(*args, **kwargs)
    return wrapper, target

class MetaProxy(torch.fx.Proxy):
    def install_tensor_meta(self, tensor_meta):
        self._tensor_meta = tensor_meta

    def size(self):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.size()
        return self.tracer.create_proxy('call_method', 'size', (self,), {})

    def dim(self):
        if hasattr(self, '_tensor_meta') and self._tensor_meta is not None:
            return self._tensor_meta.dim()
        return self.tracer.create_proxy('call_method', 'dim', (self,), {})
        
    def __getattr__(self, k):
        if k == '_tensor_meta':
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
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)

def proxys_to_metas(v):
    if isinstance(v, torch.fx.Proxy):
        assert isinstance(v, MetaProxy), f'Expected MetaProxy but got {type(v)}'
        assert hasattr(v, '_tensor_meta'), f'MetaProxy does not have an associated meta'
        return v._tensor_meta
    return v

class MetaTracer(torch.fx.Tracer):
    _TORCH_METHODS_TO_PATCH = ['arange', 'zeros', 'ones', 'full_like', 'eye']

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        if target in self.orig_fns:
            # TODO: pull out schema and also support positional args
            if 'device' in kwargs:
                kwargs['device'] = 'meta'

        try:
            if kind == 'call_function' and target != torch.fx._symbolic_trace._assert_is_none:
                args_metas = torch.fx.node.map_aggregate(args, proxys_to_metas)
                kwargs_metas = torch.fx.node.map_aggregate(kwargs, proxys_to_metas)

                meta_out = target(*args_metas, **kwargs_metas)
                assert isinstance(rv, torch.fx.Proxy), 'Dont support composite output yet'
                rv.install_tensor_meta(meta_out)
            else:
                assert kind in ['placeholder'], f'Unsupported node kind {kind}'
        except AssertionError as e:
            warnings.warn(f'Could not compute metadata for target {target}: {e}')

        return rv

    def proxy(self, node):
        return MetaProxy(node, self)

    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        root_fn, args = super().create_args_for_root(root_fn, is_module, concrete_args)
        # HACK: supporting subset of args. Do this more systematically w/ signature
        for p, meta in zip(args, self.meta_args):
            if isinstance(p, torch.fx.Proxy):
                p.install_tensor_meta(meta)
        return root_fn, args

    def trace(self, root, meta_args, concrete_args=None):
        self.meta_args = meta_args

        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            return super().trace(root, concrete_args)
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)


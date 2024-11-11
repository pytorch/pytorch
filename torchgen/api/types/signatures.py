from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, TYPE_CHECKING

from torchgen.api.types.types_base import Binding, CType, Expr


if TYPE_CHECKING:
    from torchgen.model import (
        BackendIndex,
        FunctionSchema,
        NativeFunction,
        NativeFunctionsGroup,
        NativeFunctionsViewGroup,
    )


@dataclass(frozen=True)
class CppSignature:
    """
    A CppSignature represents a single overload in the C++ API.  For
    any given function schema, there may be multiple CppSignatures
    corresponding to it, based on how we desugar to C++.  See also
    CppSignatureGroup.
    """

    # The schema this signature is derived from
    func: FunctionSchema

    # Is this a C++ signature for a method, i.e. Tensor::my_op(...)?
    method: bool

    # Is this a faithful C++ signature (i.e. following the JIT schema) or a convenience API
    # (i.e. with a potential TensorOptions argument and out arguments in the front)
    faithful: bool

    # Is this a symint C++ signature.  For BC reasons, functions that take
    # SymInts still present as int64_t in C++, and the SymInt variant is
    # offered at a different overload name
    #
    # NB: If a function RETURNS a SymInt, this is ALWAYS false
    symint: bool

    # The set of C++ arguments which should not have defaults applied to them
    cpp_no_default_args: set[str]

    # Is this a fallback C++ binding?  Fallback bindings are enabled by
    # manual_cpp_binding: True and are alternate, non-public API that
    # lets manual C++ binding implementors access the binding that would
    # have been automatically generated
    fallback_binding: bool = False

    # Return the unpacked argument structure of this signature,
    # discarding information about which arguments are semantically
    # related to each other.
    def arguments(self) -> Sequence[Binding]:
        return cpp.arguments(
            self.func.arguments,
            faithful=self.faithful,
            symint=self.symint,
            method=self.method,
            cpp_no_default_args=self.cpp_no_default_args,
        )

    def name(self, *, suppress_symint_suffix: bool = False) -> str:
        n = cpp.name(
            self.func,
            faithful_name_for_out_overloads=self.faithful,
            symint_overload=False if suppress_symint_suffix else self.symint,
        )
        if self.fallback_binding:
            n = f"__dispatch_{n}"
        return n

    # Render the C++ declaration for this signature
    def decl(
        self,
        *,
        name: str | None = None,
        prefix: str = "",
        is_redispatching_fn: bool = False,
        suppress_symint_suffix: bool = False,
    ) -> str:
        returns_type = cpp.returns_type(
            self.func.returns, symint=self.symint
        ).cpp_type()
        cpp_args = [a.decl() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ["c10::DispatchKeySet dispatchKeySet"] + cpp_args
        cpp_args_str = ", ".join(cpp_args)
        if name is None:
            name = prefix + self.name(suppress_symint_suffix=suppress_symint_suffix)
        return f"{returns_type} {name}({cpp_args_str})"

    # Render the C++ definition for this signature, not including
    # the body (with curly braces)
    def defn(
        self,
        *,
        name: str | None = None,
        prefix: str = "",
        is_redispatching_fn: bool = False,
    ) -> str:
        returns_type = cpp.returns_type(
            self.func.returns, symint=self.symint
        ).cpp_type()
        cpp_args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ["c10::DispatchKeySet dispatchKeySet"] + cpp_args
        cpp_args_str = ", ".join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    def ptr_type(self) -> str:
        args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_types_str})"

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} ({args_types_str})"


# Represents group of all CppSignatures associated with a
# FunctionSchema.  Right now, that's the regular, user-visible
# signature, as well as a "faithful" signature which doesn't
# have grouping.
@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema
    signature: CppSignature
    faithful_signature: CppSignature | None
    symint_signature: CppSignature | None
    symint_faithful_signature: CppSignature | None

    def most_faithful_signature(self) -> CppSignature:
        if self.faithful_signature:
            return self.faithful_signature
        else:
            return self.signature

    def signatures(self, *, symint: bool = True) -> Iterator[CppSignature]:
        yield self.signature
        if self.faithful_signature:
            yield self.faithful_signature
        if symint:
            if self.symint_signature:
                yield self.symint_signature
            if self.symint_faithful_signature:
                yield self.symint_faithful_signature

    @staticmethod
    def from_native_function(
        f: NativeFunction, *, method: bool, fallback_binding: bool = False
    ) -> CppSignatureGroup:
        func = f.func

        def make_sig(*, faithful: bool, symint: bool) -> CppSignature:
            return CppSignature(
                func=func,
                faithful=faithful,
                symint=symint,
                method=method,
                fallback_binding=fallback_binding,
                cpp_no_default_args=f.cpp_no_default_args,
            )

        def make_sigs(*, symint: bool) -> tuple[CppSignature, CppSignature | None]:
            faithful_signature: CppSignature | None = None
            if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
                faithful_signature = make_sig(faithful=True, symint=symint)
            signature = make_sig(faithful=False, symint=symint)
            return signature, faithful_signature

        signature, faithful_signature = make_sigs(symint=False)
        symint_signature: CppSignature | None = None
        symint_faithful_signature: CppSignature | None = None
        if func.has_symint():
            symint_signature, symint_faithful_signature = make_sigs(symint=True)

        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
            symint_signature=symint_signature,
            symint_faithful_signature=symint_faithful_signature,
        )


@dataclass(frozen=True)
class DispatcherSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # Allows you to prepend an arbitrary prefix to the signature name.
    # This is useful for parts of the codegen that generate wrappers around kernels,
    # and need to avoid naming collisions.
    prefix: str = ""

    symint: bool = True

    def arguments(self) -> list[Binding]:
        return dispatcher.arguments(self.func, symint=self.symint)

    def name(self) -> str:
        return self.prefix + dispatcher.name(self.func)

    def decl(self, name: str | None = None) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def defn(
        self, name: str | None = None, *, is_redispatching_fn: bool = False
    ) -> str:
        args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            args = ["c10::DispatchKeySet dispatchKeySet"] + args
        args_str = ", ".join(args)
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def exprs(self) -> list[Expr]:
        return [Expr(a.name, a.nctype) for a in self.arguments()]

    def returns_type(self) -> CType:
        return dispatcher.returns_type(self.func.returns, symint=self.symint)

    def ptr_type(self) -> str:
        dispatcher_args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{self.returns_type().cpp_type()} (*)({dispatcher_args_types_str})"

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        dispatcher_args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{self.returns_type().cpp_type()} ({dispatcher_args_types_str})"

    @staticmethod
    def from_schema(
        func: FunctionSchema, *, prefix: str = "", symint: bool = True
    ) -> DispatcherSignature:
        return DispatcherSignature(func, prefix, symint)


@dataclass(frozen=True)
class NativeSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    symint: bool

    prefix: str = ""

    def name(self) -> str:
        return self.prefix + native.name(self.func)

    def decl(self, name: str | None = None) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    def defn(self, name: str | None = None) -> str:
        args_str = ", ".join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    def ptr_type(self) -> str:
        # don't include defaults in type signature!
        args_str = ", ".join(a.defn() for a in self.arguments())
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_str})"

    def arguments(self) -> list[Binding]:
        return native.arguments(self.func, symint=self.symint)

    def returns_type(self) -> CType:
        return native.returns_type(self.func.returns, symint=self.symint)

    def dispatcher_exprs(self) -> list[Expr]:
        return translate.translate(
            self.arguments(), dispatcher.arguments(self.func), method=False
        )


@dataclass(frozen=True)
class ViewInverseSignature:
    g: NativeFunctionsViewGroup

    def name(self) -> str:
        return functionalization.reverse_name(self.g.view, include_namespace=False)

    def decl(self) -> str:
        return_type = functionalization.returns_type(self.g.view.func)
        decls = [
            a.decl()
            for a in functionalization.inner_arguments(
                self.g.view.func, is_reverse=True
            )
        ]
        return f"static {return_type.cpp_type()} {self.name()}({', '.join(decls)});"


@dataclass(frozen=True)
class FunctionalizationLambda:
    g: NativeFunctionsViewGroup

    # are we generating the forward lambda or the reverse lambda?
    is_reverse: bool

    def captures(self) -> list[Expr]:
        # The lambda lives inside of a kernel following the dispatcher API, so its outer context is the dispatcher arguments
        # We also need to read the "reapply views" TLS at the time that the functionalization kernel was executed,
        # and plumb it into the lambda.
        outer_ctx = dispatcher.arguments(self.g.view.func) + [
            functionalization.reapply_views_binding,
            functionalization.inverse_return_mode_binding,
        ]
        capture_bindings = functionalization.capture_arguments(
            self.g.view.func, is_reverse=self.is_reverse
        )
        # allow_expensive_conversions is set because we want to convert
        # some reference types (IntArrayRef) to value types (vector<int64_t>).
        capture_exprs = translate.translate(
            outer_ctx, capture_bindings, method=False, allow_expensive_conversions=True
        )
        return capture_exprs

    def decl(self) -> str:
        return_type = functionalization.returns_type(self.g.view.func)
        capture_str = ", ".join(
            f"{val.type.name} = {val.expr}" for val in self.captures()
        )
        decls = [
            a.decl()
            for a in functionalization.outer_arguments(is_reverse=self.is_reverse)
        ]
        return f"[{capture_str}]({', '.join(decls)}) -> {return_type.cpp_type()}"

    def inner_call(self, *, reapply_views: bool | None = None) -> str:
        inner_call_name = functionalization.name(
            self.g,
            is_reverse=self.is_reverse,
            include_namespace=True,
            reapply_views=reapply_views,
        )

        arg_ctx = functionalization.outer_arguments(is_reverse=self.is_reverse)
        capture_ctx = functionalization.capture_arguments(
            self.g.view.func, is_reverse=self.is_reverse
        )
        full_ctx = arg_ctx + capture_ctx

        assert self.g.view_copy is not None
        call_bindings = functionalization.inner_arguments(
            self.g.view_copy.func, is_reverse=self.is_reverse
        )
        maybe_index = functionalization.inner_call_index(self.g.view_copy.func)
        call_exprs = [
            e.expr for e in translate.translate(full_ctx, call_bindings, method=False)
        ]
        if not self.is_reverse and maybe_index is not None:
            return f'{inner_call_name}({", ".join(call_exprs)})[{maybe_index.name}];'
        else:
            return f'{inner_call_name}({", ".join(call_exprs)});'

    @staticmethod
    def from_func(
        g: NativeFunctionsViewGroup, *, is_reverse: bool
    ) -> FunctionalizationLambda:
        return FunctionalizationLambda(g, is_reverse)


@dataclass(frozen=True)
class StructuredImplSignature:
    g: NativeFunctionsGroup
    name: str

    def defn(self, name: str | None = None) -> str:
        args_str = ", ".join(a.defn() for a in self.arguments())
        return f"TORCH_IMPL_FUNC({self.name})({args_str})"

    def arguments(self) -> list[Binding]:
        return structured.impl_arguments(self.g)


# Helper functions


def kernel_signature(
    f: NativeFunction, backend_index: BackendIndex, *, prefix: str = ""
) -> NativeSignature | DispatcherSignature:
    # Note [External Backends Follow Dispatcher API]
    # Kernel signatures for in-tree backends follow the "native" API,
    # while kernels for out-of-tree backends follow the dispatcher API.
    # See the comments in `native.py` for details, but historically there have been
    # some small differences in schema convention between them and the Dispatcher API.
    # Any differences that require translating between the two will results in a runtime cost,
    # so we'd like to keep the differences as small as possible.
    # With external backends, we'd like to enforce that they write their kernels with schemas
    # that match the Dispatcher API directly, if they can.
    meta = backend_index.get_kernel(f)
    symint = meta is not None and meta.supports_symint()
    if symint:
        assert f.func.has_symint(), f"attempted to define symint kernel for {backend_index.dispatch_key} without SymInt in schema"
    if backend_index.external:
        return DispatcherSignature.from_schema(f.func, prefix=prefix, symint=symint)
    else:
        return NativeSignature(f.func, prefix=prefix, symint=symint)


# Functions only, no types
from torchgen.api import (
    cpp,
    dispatcher,
    functionalization,
    native,
    structured,
    translate,
)

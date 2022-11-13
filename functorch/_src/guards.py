import dataclasses
from typing import Optional
from sympy.printing.str import StrPrinter

@dataclasses.dataclass
class TensorReference(object):
    """
    TensorReference objects are entirely optional. They are created to give us hints
    into where the symbolic shape came from.

    ref_id: The id of the tensor
    kind: A string tracking where in the tensor this value came from ("size","stride", etc)
    idx: An index in the structure

    NOTE - A symbolic shape coming from tensor at id 12345's shape dim 2, would be
    TensorReference(ref_id=12345, kind="size", idx=2)
    """

    ref_id: Optional[int] = None
    kind: Optional[str] = None
    idx: Optional[int] = None
    # Note - this is untyped because of TypeError: '_SpecialForm' object does not support item assignment
    # But it is a Optional[Union["sympy.Expr", int]]
    sym_expr: Optional[object] = None  # Populated after association
    tensor_idx: Optional[int] = None

    def __hash__(self):
        return hash((self.ref_id, self.kind, self.idx))

# See: [AOT Autograd Guards Plan]
class AOTAutogradGuardPrinter(StrPrinter):
    @staticmethod
    def tensor_ref_as_str(tensor_ref, arg_list_name):
        if tensor_ref.kind in ("size", "stride"):
            return f"{arg_list_name}[{tensor_ref.tensor_idx}].{tensor_ref.kind}()[{tensor_ref.idx}]"
        return f"{arg_list_name}[{tensor_ref.tensor_idx}].{tensor_ref.kind}()"

    def __init__(
        self, expr_to_tensor_ref, arg_list_name, shape_env
    ):
        super().__init__()
        self.expr_to_tensor_ref = expr_to_tensor_ref
        self.shape_env = shape_env
        self.arg_list_name = arg_list_name

    def _print_Symbol(self, expr) -> str:
        assert isinstance(expr, sympy.core.symbol.Symbol)
        if expr == 0:
            return "0"
        if expr == 1:
            return "1"
        if expr not in self.expr_to_tensor_ref:
            return f"{self.shape_env.var_to_val[expr]}"
        # TODO(voz): Does this suffer from the same unknown symbolic issues
        # as dynamo does? So far, we have not seen it in the test suite.
        # See TORCHDYNAMO_IGNORE_ASSERT
        refs = self.expr_to_tensor_ref[expr]
        if len(refs) == 0:
            return super()._print_Symbol(expr)
        tensor_ref = next(
            iter(refs)
        )  # Any is fine here, because we install equality guards later
        return AOTAutogradGuardPrinter.tensor_ref_as_str(tensor_ref, self.arg_list_name)

# NOTE: [AOT Autograd Guards Plan]
# This produces an eval func out of guards
# Producing a string for eval is not the right way to do this going forward, 
# the right way is to decouple shape extraction from the expression set. 
# This will allow for a class of optimization where we can simplify the
# expressions agressively.
# We are keeping it eval string for now for simplicity: 
# It is very easy to reason about the correctness of a string you can read and dump
# that is just python. It also matches how dynamo operates. 
# A future refactor will:
# 1) Dedup the extraction code for tensor refs with Dynamo
# 2) Remove the production of an eval string in favor of shape info + symbolic logic
# 3) Unify the printers
def _delta_to_eval_guard_func(delta, flat_args, shape_env, arg_name):
    # We saw new guards introduced here, disjoint from the ones installed
    # upstream. We need to extract the values out and write a check
    # function
    expr_to_tensor_ref = {}
    printer = AOTAutogradGuardPrinter(expr_to_tensor_ref, arg_name, shape_env)

    # TODO(voz): Dedup with some dynamo code that is *mostly* similar
    # but differs enough around tensor_idx that its fine to keep like this for now
    # Consider fixing before landing, but okay for prototype.
    # A few differences that need to be reconciled:
    # 1) Dynamo writes to output_graph inline in its version of `_record`
    # 2) Dynamo keys on id where this keys on sym_expr because the nature of the strings
    # produced for guard eval differ
    # 3) Dymamo does not care about tensor_idx
    def extract_tensor_refs(tensor_idx, tensor):
        def _record(tensor_ref):
            if tensor_ref.sym_expr not in expr_to_tensor_ref:
                expr_to_tensor_ref[tensor_ref.sym_expr] = set()
            expr_to_tensor_ref[tensor_ref.sym_expr].add(tensor_ref)

        def _extract(symbol):
            if isinstance(symbol, int):
                return None
            sym_expr = symbol.get_pyobj().expr
            if not isinstance(sym_expr, sympy.Symbol):
                return None
            return sym_expr

        def _record_ref(e, element_index, symbol, kind, tensor_idx):
            sym_expr = _extract(symbol)
            if sym_expr:
                tensor_ref = TensorReference(id(e), kind, element_index, sym_expr, tensor_idx)
                _record(tensor_ref)

        for index, symbol in enumerate(tensor.size()):
            _record_ref(tensor, index, symbol, "size", tensor_idx)

        for index, symbol in enumerate(tensor.stride()):
            _record_ref(tensor, index, symbol, "stride", tensor_idx)

        offset = tensor.storage_offset()
        _record_ref(tensor, None, offset, "storage_offset", tensor_idx)

    for idx, tensor in enumerate(flat_args):
        extract_tensor_refs(idx, tensor)

    chained = shape_env.and_chain_guards(delta)
    return printer.doprint(chained)

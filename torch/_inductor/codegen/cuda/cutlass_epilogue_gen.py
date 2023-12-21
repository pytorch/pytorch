from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import sympy

import torch._inductor.virtualized as virtualized
from torch._inductor.codegen.cuda.cuda_template import CUTLASSTemplate
from torch._inductor.codegen.cuda.pointwise_stride_utils import (
    map_pointwise_index_to_read_strides,
)
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Layout, Pointwise
from torch._inductor.utils import IndentedBuffer


# Used as a magic string to indicate an unsupported sympy expression
# became part of generated C++ code.
_MAGIC_SYMPY_ERROR_STRING = "[!sympy: unsupported expr!]"


EVT_EXTRA_HEADER = """
namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

// Sm90AuxLoadDirect implementation ( based on Sm90ColBroadcast )
// which loads directly from global memory, without using shared memory
template<
  int _IgnoredStages,
  class CtaTileShapeMNK,
  class Element,
  class StrideMNL,
  class _IgnoredSmemLayoutAtom,
  class _IgnoredCopyOpS2R,
  int Alignment = 128 / sizeof_bits_v<Element>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct Sm90AuxLoadDirect {
  constexpr static int Stages = 0;
  static_assert(Stages == 0, "Direct load only supports 0 stages");
  static_assert(Alignment * sizeof_bits_v<Element> % 128 == 0, "sub-16B alignment not supported yet");

  // Accumulator distributes col elements evenly amongst threads so we can just directly load from gmem
  struct SharedStorage { };

  struct Arguments {
    Element const* ptr_col = nullptr;
    Element null_default = Element(0);
    StrideMNL dCol = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90AuxLoadDirect() { }

  CUTLASS_HOST_DEVICE
  Sm90AuxLoadDirect(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class GTensor, class RTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(GTensor&& tCgCol, RTensor&& tCrCol, Params const& params)
      : tCgCol(cute::forward<GTensor>(tCgCol)),
        tCrCol(cute::forward<RTensor>(tCrCol)),
        params(params) {}

    GTensor tCgCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    RTensor tCrCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    Params const& params;

    CUTLASS_DEVICE void
    begin() {
      if constexpr (EnableNullptr) {
        if (params.ptr_col == nullptr) {
          fill(tCrCol, params.null_default);
          return;
        }
      }

      // Filter so we don't issue redundant copies over stride-0 modes
      // (only works if 0-strides are in same location, which is by construction)
      copy_aligned(filter(tCgCol), filter(tCrCol));
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<Element, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<Element, FragmentSize> frg_col;
      Tensor tCrCol_mn = tCrCol(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_col[i] = tCrCol_mn(epi_v * FragmentSize + i);
      }

      return frg_col;
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    Tensor mCol = make_tensor(make_gmem_ptr(params.ptr_col), make_shape(M,N,L), params.dCol);
    Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      mCol, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrCol = make_tensor_like(tCgCol);                                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    return ConsumerStoreCallbacks<decltype(tCgCol), decltype(tCrCol)>(
      cute::move(tCgCol), cute::move(tCrCol), params);
  }
};

} // namespace cutlass::epilogue::fusion
"""


def _arg_parse(a):
    if isinstance(a, sympy.Expr):
        return a
    return str(a)


class CUTLASSEVTOpNotImplementedError(NotImplementedError):
    pass


class CutlassEVTEpilogueTypeFormatter:
    """
    Codegen class, which provides an entry point to generate
    Cutlass "Epilogue Visitor Tree" (EVT) functor declarations.

    See https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder
    for more about EVTs and how they are declared and used to generate.

    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.


    """

    def __init__(
        self,
        accumulator_node_name,
        evt_type_name,
        pre_fused_evt: Optional[str] = None,
        c_operand_alias: Optional[str] = None,
        gemm_output_layout: Optional[Layout] = None,
        flip_mn: bool = False,
        aux_load_direct: bool = True,
    ):
        """

        Initialize an instance of CutlassEVTEpilogueTypeFormatter.

        Parameters:
        - accumulator_node_name (str): The name of the output Buffer for the GEMM operation in the original (unfused)
                                       IR graph.
        - evt_type_name (str):      The output name of the EVT type we are generating.
        - pre_fused_evt (Optional[str]): Optional EVT expression declaration that is pre-fused into the template
                                    (typically addmm style bias addition etc.)
        - c_operand_alias (Optional[str]): Optional name of the C operand
        - gemm_output_layout: Output layout of the GEMM operation.
        - flip_mn: Whether to flip the M and N dimensions of the GEMM output layout.
        - aux_load_direct: Whether to use direct loads for auxiliary inputs (i.e. load directly from global memory)

        """
        self.accumulator_node_name = accumulator_node_name
        self.output = IndentedBuffer(0)
        self.var_counter = 0
        self.evt_type_name = evt_type_name
        self.aliases: Dict[str, Optional[str]] = dict()
        self.pre_fused_evt = pre_fused_evt
        self.c_operand_alias = c_operand_alias
        self.accumulator_index_expr = None
        self.gemm_output_layout = gemm_output_layout
        self.flip_mn = flip_mn
        self.aux_load_direct = aux_load_direct

    @staticmethod
    def ir_to_evt_string(
        template_output_node_name: str,
        evt_type_name: str,
        epilogue_nodes: List[IRNode],
        pre_fused_evt: Optional[str] = None,
        c_operand_alias: Optional[str] = None,
        gemm_output_layout: Optional[Layout] = None,
        flip_mn: bool = False,
        aux_load_direct: bool = True,
    ):
        """
        Formats IR nodes into a string representation compatible with Cutlass EVT format.

        Args:
            template_output_node_name (str): The name of the template output node.
            evt_type_name (str): The name of the EVT type.
            epilogue_nodes (List[IRNode]): A list of IR nodes representing the epilogue nodes. As of now, these must be
                ComputedBuffer nodes wrapping Pointwise nodes.
            pre_fused_evt: Optional EVT expression declaration that is pre-fused into the template
                           (typically addmm style bias addition etc.)
            c_operand_alias: Optional name of the C operand
            gemm_output_layout: Output layout of the GEMM operation.
            flip_mn: Whether to flip the M and N dimensions of the GEMM output layout.
            aux_load_direct: Whether to use direct loads for auxiliary inputs (i.e. load directly from global memory)

        Returns:
            A string representation of the IR nodes formatted according to the Cutlass EVT format.
        """
        if epilogue_nodes is None:
            epilogue_nodes = []
        if pre_fused_evt is None and len(epilogue_nodes) == 0:
            return f"using {evt_type_name} = cutlass::epilogue::fusion::Sm90AccFetch"

        formatter = CutlassEVTEpilogueTypeFormatter(
            template_output_node_name,
            evt_type_name,
            pre_fused_evt,
            c_operand_alias,
            gemm_output_layout,
            flip_mn,
            aux_load_direct,
        )

        with virtualized.V.set_ops_handler(formatter), patch.object(  # type: ignore[call-arg]
            FlexibleLayout, "allow_indexing", True
        ):
            if pre_fused_evt is not None:
                result = formatter.pre_fused_expr(pre_fused_evt)
            for node in epilogue_nodes:
                if isinstance(node, ComputedBuffer):
                    pnode = node.data
                else:
                    raise RuntimeError(
                        "Epilogue nodes must be Pointwise nodes, wrapped in a named ComputedBuffer"
                    )
                assert isinstance(pnode, Pointwise)
                index = pnode._index(pnode.ranges)

                CutlassEVTEpilogueTypeFormatter.check_range_and_stride_compatibility(
                    gemm_output_layout, pnode
                )

                formatter.current_index_symbols = index  # type: ignore[attr-defined]
                result = pnode.inner_fn(index)
                # each epilogue node results in a single "using" statement and may refer to the previous steps by name
                if node.name is not None:
                    formatter.aliases[node.name] = result
            res = formatter.getvalue(result)
            if _MAGIC_SYMPY_ERROR_STRING in res:
                raise CUTLASSEVTOpNotImplementedError(
                    "sympy / indexing expressions not yet supported in EVT fusion"
                )
            else:
                return res

    @staticmethod
    def check_range_and_stride_compatibility(gemm_output_layout, pnode):
        if len(pnode.ranges) == len(gemm_output_layout.stride):
            return
        if not gemm_output_layout.is_contiguous():
            raise CUTLASSEVTOpNotImplementedError(
                f"For non-contiguous tensors, Pointwise.ranges {pnode.ranges} must have the same length as the strides of the GEMM output layout {gemm_output_layout}. Pointwise op: {pnode}"  # noqa: B950
            )
        if len(pnode.ranges) > len(gemm_output_layout.stride):
            raise CUTLASSEVTOpNotImplementedError(
                f"For non-contiguous tensors, Pointwise.ranges {pnode.ranges} must have the same length as the strides of the GEMM output layout {gemm_output_layout}. Pointwise op: {pnode}"  # noqa: B950
            )
        return

    @staticmethod
    def create_pre_fused_addmm_evt_type() -> str:
        """returns the name of the ADDMM EVT type which has been declared like this:

        using ADDMM_EVT =  // alpha * acc + beta * C
            cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add,
                        ElementD, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
                  cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // beta
                  cutlass::epilogue::fusion::Sm90SrcFetch, // C
                  cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,
                        ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
                    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // alpha
                    cutlass::epilogue::fusion::Sm90AccFetch // acc
              >>
        """
        return "ADDMM_EVT"

    def __getattr__(self, name):
        """
        Resolve V.ops.<whatever> calls, after this instance has been installed as V.ops handler.
        """

        def inner(*args, **kwargs):
            fargs = [_arg_parse(a) for a in args]
            fkwargs = {key: _arg_parse(a) for key, a in kwargs.items()}
            fn = getattr(self, f"_op_{name}")
            line = fn(*fargs, **fkwargs)
            self.var_counter += 1
            varname = f"EVT_expr_{self.var_counter}"
            # replace line with a new variable name
            self.output.writeline(f"using {varname} = {line};")
            return varname

        if name.startswith("_"):
            raise CUTLASSEVTOpNotImplementedError(name)
        if hasattr(self, f"_op_{name}"):
            return inner
        else:
            raise CUTLASSEVTOpNotImplementedError(name)

    def _aux_load_decl(self, name, index_expr):
        graph = virtualized.V.graph
        node = graph.get_buffer(name)
        assert (
            node is not None
        ), f"Input buffer with name {name} not found in current graph"
        aux_load_descriptor, strides_mnl = create_cutlass_aux_load_descriptor(
            node,
            index_expr,
            self.accumulator_index_expr,
            self.gemm_output_layout,
            self.flip_mn,
        )
        ALD = f"{name}AuxLoadDesc"
        self.output.writeline(f"using {ALD} = {aux_load_descriptor};")
        if strides_mnl == (0, 1, 0):
            # Special case: row broadcast
            aux_load_op = "Sm90RowBroadcast"
            aux_load_template_args = f"{ALD}::Stages, TileShapeMNK, typename {ALD}::Element, typename {ALD}::Stride"
        elif strides_mnl == (1, 0, 0):
            # Special case: col broadcast
            aux_load_op = "Sm90ColBroadcast"
            # ColBroadcast only supports 0 stages, since it doesn't use smem
            aux_load_template_args = (
                f"0, TileShapeMNK, typename {ALD}::Element, typename {ALD}::Stride"
            )
        elif (
            strides_mnl in ((0, 0, 0), (1, 1, 0), (0, 0, 1), (1, 1, 1))
            or (strides_mnl[1] == 0 and strides_mnl[0] > 1)
            or (strides_mnl[0] == 0 and strides_mnl[1] > 1)
        ):
            raise CUTLASSEVTOpNotImplementedError(
                f"Unsupported broadcast strides for aux input {name}: {strides_mnl=}"
            )
        else:
            if self.aux_load_direct:
                aux_load_op = "Sm90AuxLoadDirect"
            else:
                aux_load_op = "Sm90AuxLoad"
            aux_load_template_args = f"{ALD}::Stages, TileShapeMNK, typename {ALD}::Element, typename {ALD}::Stride, typename {ALD}::SmemLayoutAtom, typename {ALD}::CopyOpS2R"  # noqa: B950

        return f"""cutlass::epilogue::fusion::Sm90EVT<
                                        cutlass::epilogue::fusion::Sm90Compute<identity_op,ElementAcc, typename {ALD}::Element, RoundStyle >,
                                        cutlass::epilogue::fusion::{aux_load_op}<{aux_load_template_args}>> /* :={name} as aux operand, cast to accumulator dtype */"""  # noqa: B950

    def _op_load(self, name, index_expr):
        # Load an input to an operation. Might be the output of the matmul, the result
        # of a previous epilogue node, a constant or (TODO) an auxiliary input.
        if name == self.accumulator_node_name:
            self.accumulator_index_expr = index_expr
            if self.pre_fused_evt is None:
                return f"cutlass::epilogue::fusion::Sm90AccFetch /* :={name} (matmul output in accumulator) */"
            else:
                return self.pre_fused_evt
        elif name in self.aliases:
            return self.aliases[name]
        elif name == self.c_operand_alias:
            return f"""cutlass::epilogue::fusion::Sm90EVT<
                                cutlass::epilogue::fusion::Sm90Compute<identity_op,ElementAcc, ElementC, RoundStyle >,
                                cutlass::epilogue::fusion::Sm90SrcFetch> /* :={name} as operand C, cast to accumulator dtype */"""
        else:
            return self._aux_load_decl(name, index_expr)

    def _op_constant(self, value, dtype):
        # Load a constant
        if str(dtype) in ("torch.float16", "torch.float32"):
            return f"cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAcc> /* value={value}, dtype={dtype} */"
        else:
            raise CUTLASSEVTOpNotImplementedError(
                f"Unsupported dtype for constant: {dtype}"
            )

    def _cutlass_binary_functional_op(self, op, a, b):
        # Perform a named operation on two inputs
        # see https://github.com/NVIDIA/cutlass/blob/6407bcdf0a24097b7b016ee105937693c62f9923/include/cutlass/functional.h for ops
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::{op}, ElementAcc, ElementAcc, RoundStyle>,{a},{b}>"  # noqa: B950

    def _convert_to_output_dtype(self, a):
        # Convert the final output to the dtype of the output buffer
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<identity_op, ElementD, ElementAcc, RoundStyle>,{a}>"  # noqa: B950

    def _op_to_dtype(self, a, *args, **kwargs):
        # no-op in our case, since we convert to the output dtype at the end and convert everything to the accumulator
        # dtype.
        # Is is asserted ( and ascertained during can_fuse decision ) that the dtype remains compatible
        # throughout the fusion chain.
        return a  # noqa: B950

    def _op_mul(self, a, b):
        return self._cutlass_binary_functional_op("multiplies", a, b)

    def _op_div(self, a, b):
        return self._cutlass_binary_functional_op("divides", a, b)

    def _op_truediv(self, a, b):
        return self._cutlass_binary_functional_op("divides", a, b)

    def _op_ge(self, a, b):
        return self._cutlass_binary_functional_op("greater_equal", a, b)

    def _op_add(self, a, b):
        return self._cutlass_binary_functional_op("plus", a, b)

    def _op_sub(self, a, b):
        return self._cutlass_binary_functional_op("minus", a, b)

    def _op_minimum(self, a, b):
        return self._cutlass_binary_functional_op("minimum", a, b)

    def _op_maximum(self, a, b):
        return self._cutlass_binary_functional_op("maximum", a, b)

    def _op_relu(self, a):
        const_zero = self._op_constant(0.0, "torch.float32")
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::maximum, ElementAcc, ElementAcc, RoundStyle>,{a}, {const_zero}>"  # noqa: B950

    def _op_sigmoid(self, a):
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Sigmoid, ElementAcc, ElementAcc, RoundStyle>,{a}>"  # noqa: B950

    def _op_tanh(self, a):
        return f"cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Tanh, ElementAcc, ElementAcc, RoundStyle>,{a}>"  # noqa: B950

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise CUTLASSEVTOpNotImplementedError()

    # Add more ops here...
    def getvalue(self, result) -> str:
        # Return final result
        dtype_converted_expr = self._convert_to_output_dtype(
            f"EVT_expr_{self.var_counter}"
        )
        self.output.writeline(f"using {self.evt_type_name} = {dtype_converted_expr};")
        return self.output.getvalue()

    def _op_pre_fused_expr(self, expr):
        return expr


class CutlassEVTEpilogueArgumentFormatter:
    """
    Codegen class, which provides an entry point to generate
    Cutlass "Epilogue Visitor Tree" (EVT) Argument initializers

    See https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder
    for more about EVTs and how they are declared and used to generate.

    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.


    """

    def __init__(
        self,
        accumulator_node_name: str,
        pre_fused_evt_args: Optional[str] = None,
        c_operand_alias: Optional[str] = None,
        dry_run: bool = False,
        gemm_output_layout: Optional[Layout] = None,
        flip_mn: bool = False,
    ):
        """

        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string static method.

        Args:
            accumulator_node_name (str): The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
            pre_fused_evt_args (Optional[str]): Optional arguments for a pre-fused EVT expression (typically addmm args).
            dry_run(bool): If true, will not require an actual Kernel as context and assume we're only doing validity checking
            gemm_output_layout: Output layout of the GEMM operation.
        """
        self.accumulator_node_name: str = accumulator_node_name  #
        self.output: IndentedBuffer = IndentedBuffer(0)  # The output buffer for codegen
        self.var_counter: int = (
            0  # used to generate variable names, incremented for each new variable
        )
        self.pre_fused_evt_args: Optional[str] = pre_fused_evt_args
        self.aliases: Dict[str, str] = dict()  # Aliases for subexpression functors
        self.c_operand_alias = c_operand_alias
        self.dry_run = dry_run
        self.gemm_output_layout = gemm_output_layout
        self.flip_mn = flip_mn

    @staticmethod
    def ir_to_evt_argument_string(
        template_output_node_name: str,
        epilogue_nodes: List[IRNode],
        pre_fused_evt_args: Optional[str] = None,
        c_operand_alias: Optional[str] = None,
        dry_run: bool = False,
        gemm_output_layout: Optional[Layout] = None,
        flip_mn: bool = False,
    ) -> str:
        formatter = CutlassEVTEpilogueArgumentFormatter(
            template_output_node_name,
            pre_fused_evt_args,
            c_operand_alias,
            dry_run,
            gemm_output_layout,
            flip_mn,
        )
        result = pre_fused_evt_args
        if (pre_fused_evt_args is None) and (
            (epilogue_nodes is None) or len(epilogue_nodes) == 0
        ):
            return "{}"
        with virtualized.V.set_ops_handler(formatter), patch.object(  # type: ignore[call-arg]
            FlexibleLayout, "allow_indexing", True
        ):
            for node in epilogue_nodes:
                assert isinstance(node, ComputedBuffer)
                pnode = node.data
                assert isinstance(pnode, Pointwise)
                assert gemm_output_layout is not None  # for mypy
                assert len(pnode.ranges) == len(gemm_output_layout.stride)
                index = pnode._index(pnode.ranges)
                result = pnode.inner_fn(index)
                # each epilogue node results in a single "using" statement and may refer to the previous steps by name
                if node.name is not None and result is not None:
                    formatter.aliases[node.name] = result  # type: ignore[assignment]

            res: str = formatter.getvalue(result)  # type: ignore[possibly-undefined]
            if _MAGIC_SYMPY_ERROR_STRING in res:
                raise CUTLASSEVTOpNotImplementedError(
                    "sympy / indexing expressions not yet supported in EVT fusion"
                )
            else:
                return res

    @staticmethod
    def create_pre_fused_addmm_arg_str(alpha: float, beta: float) -> str:
        return """
        {  // ADDMM Arguments: ternary op : beta * C + (alpha * acc)
          {{static_cast<ElementAcc>(%f)}}, // leaf op+args : beta
          {},               // leaf op+args : C
          {                 // binary op : alpha * acc
            {{static_cast<ElementAcc>(%f)}}, // leaf op+args : alpha
            {},                // leaf op+args : acc
            {}              // binary args : multiplies
          },                // end binary op
          {} // ternary args : multiply_add
        }   // end ternary op
        """ % (  # noqa: UP031
            beta,
            alpha,
        )

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            fargs = [_arg_parse(a) for a in args]
            fkwargs = {key: _arg_parse(a) for key, a in kwargs.items()}
            fn = getattr(self, f"_op_{name}")
            line = fn(*fargs, **fkwargs)
            return line

        if name.startswith("_"):
            raise CUTLASSEVTOpNotImplementedError(name)

        if hasattr(self, f"_op_{name}"):
            return inner
        else:
            raise CUTLASSEVTOpNotImplementedError(name)

    def _op_load(self, name, index_expr):
        if name == self.accumulator_node_name:
            if self.pre_fused_evt_args is None:
                return "{}"
            else:
                return self.pre_fused_evt_args
        elif name in self.aliases:
            return self.aliases[name]
        elif name == self.c_operand_alias:
            return "{}"
        else:
            if self.dry_run:
                return f"{{ /* dry run placeholder for aux input {name} */ }}"
            kernel = virtualized.V.kernel
            from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateKernel

            assert isinstance(kernel, CUDATemplateKernel)
            aux_arg_name = kernel.args.input_buffers[name]
            assert (
                aux_arg_name in kernel.named_nodes
            ), f"Auxiliary argument {aux_arg_name} not found in kernel"
            aux_input_node = kernel.named_nodes[aux_arg_name]
            cutlass_dtype = CUTLASSTemplate._DTYPE_TO_CUTLASS[
                aux_input_node.get_layout().dtype
            ]
            # cpp_dtype = kernel.dtype(aux_input_node)
            data_ptr = kernel.ptr(aux_input_node)
            assert self.gemm_output_layout is not None  # for mypy
            strides: List[int] = map_pointwise_index_to_read_strides(
                index_expr, self.gemm_output_layout, self.flip_mn
            )
            # For the sanity check,
            # output size dimensions need to be flipped if flip_mn=True for the GEMM
            # since the output layout (incl. sizes) might have been flipped
            # from what is actually written
            gemm_write_sizes = list(self.gemm_output_layout.size)
            if self.flip_mn:
                gemm_write_sizes[-1], gemm_write_sizes[-2] = (
                    gemm_write_sizes[-2],
                    gemm_write_sizes[-1],
                )

            load_stride_max_idx = sum(
                [stride * (dim - 1) for stride, dim in zip(strides, gemm_write_sizes)]
            )
            # The strides might have reinterpreted the buffer, but they may not read beyond it's bounds, let's check that...
            assert (
                load_stride_max_idx < aux_input_node.get_numel()
            ), f"Aux input would read beyond bounds (A): Load stride {strides} for node {aux_input_node.get_name()} with layout {aux_input_node.get_layout()} - accessed using index expr {index_expr} is too large for the node when mapped onto GEMM with output layout {self.gemm_output_layout}."  # noqa: B950
            if len(strides) == 2:
                strides.insert(0, 0)

            stride_strings = [
                f"{stride}L" if stride not in [0, 1] else f"cute::Int<{stride}L>{{}}"
                for stride in strides
            ]
            stride_strings = [stride_strings[-2], stride_strings[-1], stride_strings[0]]
            stride_args = ", ".join(stride_strings)
            return f"""{{ (({cutlass_dtype}*)({data_ptr})), {cutlass_dtype}(0), {{ {stride_args} }} }} /* {name} data pointer incl. offset, zero element value and strides for MNL (L=batch) dims */"""  # noqa: B950

    def _op_constant(self, value, dtype):
        if str(dtype) in ("torch.float16", "torch.float32"):
            return "{ static_cast<ElementAcc>(" + str(value) + ") }"
        else:
            raise CUTLASSEVTOpNotImplementedError(
                f"Unsupported dtype for constant: {dtype}"
            )

    def _cutlass_binary_functional_op(self, op, a, b):
        return f"{{ /*{op}: */ {a}, {b} }}"

    def _op_mul(self, a, b):
        return self._cutlass_binary_functional_op("multiplies", a, b)

    def _op_div(self, a, b):
        return self._cutlass_binary_functional_op("divides", a, b)

    def _op_truediv(self, a, b):
        return self._cutlass_binary_functional_op("divides", a, b)

    def _op_ge(self, a, b):
        return self._cutlass_binary_functional_op("greater_equal", a, b)

    def _op_add(self, a, b):
        return self._cutlass_binary_functional_op("plus", a, b)

    def _op_sub(self, a, b):
        return self._cutlass_binary_functional_op("minus", a, b)

    def _op_minimum(self, a, b):
        return self._cutlass_binary_functional_op("minimum", a, b)

    def _op_maximum(self, a, b):
        return self._cutlass_binary_functional_op("maximum", a, b)

    def _op_relu(self, a):
        const_zero = self._op_constant(0.0, "torch.float32")
        return "{" + str(a) + ", " + const_zero + "}"

    def _op_sigmoid(self, a):
        return "{}"

    def _op_tanh(self, a):
        return "{}"

    def _op_to_dtype(self, a, dtype, src_dtype=None):
        # Is is asserted ( and ascertained during can_fuse decision ) that the dtype remains compatible
        # throughout the fusion chain.
        assert dtype in (
            "torch.float32",
            "torch.float16",
        ), f"Unsupported dtype: {dtype}"
        assert src_dtype in (
            None,
            "torch.float32",
            "torch.float16",
        ), f"Unsupported source dtype: {src_dtype}"
        return a

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise CUTLASSEVTOpNotImplementedError()

    def getvalue(self, result) -> str:
        return "{" + str(result) + "}"


def cute_stride_decl(strides, stride_dtype: str = "int64_t"):
    stride_args = []
    for stride in strides:
        if str(stride) in ["0", "1", "0L", "1L"]:
            stride_args.append(f"cute::Int<{stride}>")
        else:
            stride_args.append(stride_dtype)
    return "cute::Stride<" + ", ".join(stride_args) + ">"


def cute_stride_mnl_decl(strides):
    if len(strides) >= 3:
        mnl_strides = [strides[-2], strides[-1]] + list(strides[:-2][::-1])
    else:
        mnl_strides = [strides[-2], strides[-1]] + [0]
    return cute_stride_decl(mnl_strides), tuple(mnl_strides)


def create_cutlass_aux_load_descriptor(
    node: IRNode, index_expr, accumulator_index_expr, gemm_output_layout, flip_mn=False
) -> Tuple[str, Tuple[int, int, int]]:
    """
    Creates a Cutlass auxiliary descriptor for the given node.
    This is used to pass auxiliary inputs to the kernel.
    """
    cutlass_dtype = CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]
    # these load strides are extracted from the index expression, and therefore properly mapped to the output layout.
    strides: List[int] = map_pointwise_index_to_read_strides(
        index_expr, gemm_output_layout, flip_mn
    )
    # For the sanity check,
    # output size dimensions need to be flipped if flip_mn=True for the GEMM
    # since the output layout (incl. sizes) might have been flipped
    # from what is actually written
    gemm_write_sizes = list(gemm_output_layout.size)
    if flip_mn:
        gemm_write_sizes[-1], gemm_write_sizes[-2] = (
            gemm_write_sizes[-2],
            gemm_write_sizes[-1],
        )
    load_stride_max_idx = sum(
        [stride * (dim - 1) for stride, dim in zip(strides, gemm_write_sizes)]
    )
    # The strides might have reinterpreted the buffer, but they may not read beyond it's bounds, let's check that...
    assert (
        load_stride_max_idx < node.get_numel()
    ), f"Aux input would read beyond bounds (B): Load stride {strides} for node {node.get_name()} with layout {node.get_layout()} - accessed using index expr {index_expr} is too large for the node when mapped onto GEMM with output layout {gemm_output_layout}."  # noqa: B950
    if len(strides) < 2:
        raise CUTLASSEVTOpNotImplementedError(
            f"Unsupported number of strides: {len(strides)} for aux input for node {node.get_name()} with layout {node.get_layout()} - accessed using index expr {index_expr}. Accumulator index expr={accumulator_index_expr}"  # noqa: B950
        )
    layout_stride_decl, mnl_strides = cute_stride_mnl_decl(strides)
    return (
        f"""cutlass::epilogue::collective::detail::AuxLoadDescriptor<EpilogueDescriptor, {layout_stride_decl}, {cutlass_dtype}>""",  # noqa: B950
        mnl_strides,
    )

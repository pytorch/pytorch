# Owner(s): ["module: inductor"]
import unittest

import sympy

import torch
from torch._dynamo.test_case import TestCase
from torch._inductor.codegen.cuda.cutlass_utils import (
    torch_dtype_to_cutlass_type,
    try_import_cutlass,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FixedLayout, PermuteView, Pointwise
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import OrderedSet
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


if try_import_cutlass():
    import cutlass_library as cutlass_lib
    from cutlass_library import EpilogueScheduleType

    LayoutType = cutlass_lib.LayoutType
    DataType = cutlass_lib.DataType
    from torch._inductor.codegen.cuda.cutlass_lib_extensions.evt_extensions import (
        _render_argument_type,
        _trace,
        CutlassTensor,
        trace,
    )

    BIAS_CODE = """def example_epilogue(accum, C, aux, bias):
        F = accum + C + aux
        E = relu(F) + bias
        D = E + F
        return D, F"""

    TYPE_C = DataType.f32
    M = 4224
    N = 2048
    BIAS = CutlassTensor(shape=(M, 1), element=TYPE_C, layout_tag=LayoutType.RowMajor)

    EXAMPLE_TENSORS = {
        "accum": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
        "bias": BIAS,
        # "beta": 0.5, TODO: mlazos support scalars
        # "alpha": 0.5, TODO: mlazos support scalars
        "D": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
        "C": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
        "F": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
        "aux": CutlassTensor(
            element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
        ),
    }

    class MockTileDescription:
        threadblock_shape = (128, 128, 8)

    def _create_mock_buffer_name_map(example_tensors):
        name_to_buffer = {}
        for name, tensor in example_tensors.items():
            if isinstance(tensor, CutlassTensor):
                name_to_buffer[name] = MockComputedBuffer(
                    name, None, torch.float32, tensor.shape, tensor.stride
                )

        return name_to_buffer


class MockSchedulerNode(BaseSchedulerNode):
    def __init__(self, node, last_usage=None):
        self.node = node
        self.last_usage = last_usage or OrderedSet()


class MockComputedBuffer(ComputedBuffer):
    def __init__(self, name, inner_fn, dtype, size, strides=None):
        self.name = name
        ranges = [sympy.Integer(x) for x in size]
        self.data = Pointwise(
            device=None, dtype=dtype, inner_fn=inner_fn, ranges=ranges
        )
        self.layout = FixedLayout(None, dtype, ranges, strides)

    def get_name(self):
        return self.name

    def num_reads(self):
        # Needed to not inline in ComputedBuffer
        return 1


class MockGraphHandler(GraphLowering):
    def __init__(self, name_to_buffer):
        import torch._inductor.sizevars

        self.sizevars = torch._inductor.sizevars.SizeVarAllocator()
        self.name_to_buffer = name_to_buffer
        self.graph_inputs = dict()
        self.mutated_buffers = OrderedSet()
        self.constants = dict()


class TestCutlassEVT(TestCase):
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_accumulator_return(self):
        from torch._inductor.codegen.cuda.cutlass_python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)
        buf1 = MockComputedBuffer("buf1", None, torch.float32, size)
        buf2 = MockComputedBuffer("buf2", None, torch.float32, size)

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            tmp0 = buf0.make_loader()(index)
            tmp1 = buf1.make_loader()(index)
            tmp2 = buf2.make_loader()(index)
            return tmp0 * tmp1 + tmp2

        def inner_fn_buf4(index):
            tmp0 = buf0.make_loader()(index)
            tmp3 = buf3.make_loader()(index)
            return tmp0 + tmp3

        buf3 = MockComputedBuffer("buf3", inner_fn_buf3, torch.float32, size)
        buf4 = MockComputedBuffer("buf4", inner_fn_buf4, torch.float32, size)
        with V.set_graph_handler(
            MockGraphHandler(
                {"buf0": buf0, "buf1": buf1, "buf2": buf2, "buf3": buf3, "buf4": buf4}
            )
        ):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [
                    MockSchedulerNode(buf3),
                    MockSchedulerNode(buf4, last_usage=OrderedSet(["buf3"])),
                ],
                OrderedSet([]),
            )
        self.assertExpectedInline(reads, """['buf1', 'buf2']""")
        self.assertExpectedInline(writes, """['buf0', 'buf3', 'buf4']""")
        self.assertExpectedInline(
            renames,
            """{'accum': 'buf0', 'tmp_0': 'buf0', 'buf1': 'buf1', 'buf2': 'buf2', 'tmp_2': 'buf3', 'D': 'buf4'}""",
        )
        self.assertExpectedInline(
            code,
            """\
def fn(accum, buf1, buf2):
    tmp_0 = accum
    tmp_1 = tmp_0 * buf1
    tmp_2 = tmp_1 + buf2
    D = tmp_0 + tmp_2

return tmp_0, tmp_2, D""",
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_disjoint_read_indexing(self):
        from torch._inductor.codegen.cuda.cutlass_python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)
        permuted_buf_0 = PermuteView.create(buf0, [1, 0, 2])
        buf1 = MockComputedBuffer("buf1", None, torch.float32, size)
        buf2 = MockComputedBuffer("buf2", None, torch.float32, size)

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            tmp0 = permuted_buf_0.make_loader()(index)
            tmp1 = buf1.make_loader()(index)
            tmp2 = buf2.make_loader()(index)
            return tmp0 * tmp1 + tmp2

        def inner_fn_buf4(index):
            tmp0 = buf0.make_loader()(index)
            tmp3 = buf3.make_loader()(index)
            return tmp0 + tmp3

        buf3 = MockComputedBuffer("buf3", inner_fn_buf3, torch.float32, size)
        buf4 = MockComputedBuffer("buf4", inner_fn_buf4, torch.float32, size)

        with V.set_graph_handler(
            MockGraphHandler(
                {"buf0": buf0, "buf1": buf1, "buf2": buf2, "buf3": buf3, "buf4": buf4}
            )
        ):
            result = None
            try:
                CutlassEVTCodegen.ir_to_evt_python_code(
                    "buf0",
                    [MockSchedulerNode(buf3), MockSchedulerNode(buf4)],
                    OrderedSet([]),
                )
            except NotImplementedError as e:
                result = e

            self.assertExpectedInline(
                str(result),
                """Unsupported indexing for buf0 with index 200*i0 + 60000*i1 + i2, \
index strides [200, 60000, 1], and layout stride [60000, 200, 1]""",
            )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen_broadcasting(self):
        from torch._inductor.codegen.cuda.cutlass_python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)
        buf1 = MockComputedBuffer("buf1", None, torch.float32, size)
        buf2 = MockComputedBuffer("buf2", None, torch.float32, size)

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            tmp0 = buf0.make_loader()(index)
            tmp1 = buf1.make_loader()(index)
            tmp2 = buf2.make_loader()(index)
            return tmp0 * tmp1 + tmp2

        def inner_fn_buf4(index):
            tmp0 = buf0.make_loader()(index)
            tmp3 = buf3.make_loader()(index)
            return tmp0 + tmp3 * tmp3

        buf3 = MockComputedBuffer("buf3", inner_fn_buf3, torch.float32, size)
        buf4 = MockComputedBuffer(
            "buf4", inner_fn_buf4, torch.float32, (100, 300, 1)
        )  # broadcast
        with V.set_graph_handler(
            MockGraphHandler(
                {"buf0": buf0, "buf1": buf1, "buf2": buf2, "buf3": buf3, "buf4": buf4}
            )
        ):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [
                    MockSchedulerNode(buf3),
                    MockSchedulerNode(buf4, last_usage=OrderedSet(["buf0"])),
                ],
                OrderedSet([]),
            )
        self.assertExpectedInline(reads, """['buf1', 'buf2']""")
        self.assertExpectedInline(writes, """['buf0', 'buf3', 'buf4']""")
        self.assertExpectedInline(
            renames,
            """{'accum': 'buf0', 'tmp_0': 'buf0', 'buf1': 'buf1', 'buf2': 'buf2', 'tmp_2': 'buf3', 'D': 'buf4'}""",
        )
        self.assertExpectedInline(
            code,
            """\
def fn(accum, buf1, buf2):
    tmp_0 = accum
    tmp_1 = tmp_0 * buf1
    tmp_2 = tmp_1 + buf2
    tmp_3 = tmp_2 * tmp_2
    D = tmp_0 + tmp_3

return tmp_0, tmp_2, D""",
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_py_codegen(self):
        from torch._inductor.codegen.cuda.cutlass_python_evt import CutlassEVTCodegen
        from torch._inductor.virtualized import V

        size = (100, 300, 200)
        buf0 = MockComputedBuffer("buf0", None, torch.float32, size)
        buf1 = MockComputedBuffer("buf1", None, torch.float32, size)
        buf2 = MockComputedBuffer("buf2", None, torch.float32, size)

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            tmp0 = buf0.make_loader()(index)
            tmp1 = buf1.make_loader()(index)
            tmp2 = buf2.make_loader()(index)
            return tmp0 * tmp1 + tmp2

        def inner_fn_buf4(index):
            tmp0 = buf0.make_loader()(index)
            tmp3 = buf3.make_loader()(index)
            return tmp0 + tmp3

        buf3 = MockComputedBuffer("buf3", inner_fn_buf3, torch.float32, size)
        buf4 = MockComputedBuffer("buf4", inner_fn_buf4, torch.float32, size)
        with V.set_graph_handler(
            MockGraphHandler(
                {"buf0": buf0, "buf1": buf1, "buf2": buf2, "buf3": buf3, "buf4": buf4}
            )
        ):
            reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
                "buf0",
                [
                    MockSchedulerNode(buf3),
                    MockSchedulerNode(buf4),
                ],
                OrderedSet(["buf0"]),
            )
        self.assertExpectedInline(reads, """['buf1', 'buf2']""")
        self.assertExpectedInline(writes, """['buf3', 'buf4']""")
        self.assertExpectedInline(
            renames,
            """{'accum': 'buf0', 'buf1': 'buf1', 'buf2': 'buf2', 'tmp_1': 'buf3', 'D': 'buf4'}""",
        )
        self.assertExpectedInline(
            code,
            """\
def fn(accum, buf1, buf2):
    tmp_0 = accum * buf1
    tmp_1 = tmp_0 + buf2
    D = accum + tmp_1

return tmp_1, D""",
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_example_tensor_creation(self):
        from torch._inductor.codegen.cuda.cutlass_lib_extensions.evt_extensions import (
            create_example_tensors,
        )

        row_major_buf0 = MockComputedBuffer(
            "buf0", None, torch.float32, (3, 4, 1), (4, 1, 0)
        )
        col_major_buf1 = MockComputedBuffer(
            "buf1", None, torch.float32, (3, 2, 1), (1, 3, 0)
        )
        buffer_renames = {"buf0": "buf0", "buf1": "buf1", "acc": "buf0"}
        name_to_buffer = {"buf0": row_major_buf0, "buf1": col_major_buf1}
        result = create_example_tensors(
            buffer_renames, name_to_buffer, lambda x: int(x)
        )
        self.assertEqual(result["acc"].shape, (3, 4, 1))
        self.assertEqual(result["acc"].stride, (4, 1, 0))
        self.assertEqual(
            result["acc"].element, torch_dtype_to_cutlass_type(torch.float32)
        )

        self.assertEqual(result["buf1"].shape, (3, 2, 1))
        self.assertEqual(result["buf1"].stride, (1, 3, 0))
        self.assertEqual(
            result["buf1"].element, torch_dtype_to_cutlass_type(torch.float32)
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_evt_argument_codegen(self):
        from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch

        cuda_arch = int(get_cuda_arch())  # type: ignore[arg-type]
        epilogue_functor = _trace(BIAS_CODE, EXAMPLE_TENSORS, cuda_arch)

        self.assertExpectedInline(
            _render_argument_type(
                epilogue_functor,
                _create_mock_buffer_name_map(EXAMPLE_TENSORS),
                lambda x: int(x),
            ),
            """\
{ /* thread */
        { /* F */
          { /* compute_1 */
            { /* compute_0 */
              {}, /* accum */
              {}, /* C */
              {}, /* compute_0 */
            },
            {/* ptr_aux */ (float*) aux, /* null_default */ float(0), /* dAux */ {2048, _1{}, _0{}}}, /* aux */
            {}, /* compute_1 */
          },
          {/* ptr_aux */ (float*) F, /* dAux */ {2048, _1{}, _0{}}}, /* F */
        },
        {/* ptr_col */ (float*) bias, /* null_default */ float(0), /* dCol */ {}}, /* bias */
        {}, /* compute_2 */
        {}, /* compute_3 */
        {}, /* compute_4 */
      }
""",
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_evt_argument_codegen_return_accumulator(self):
        from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch

        code = """
def fn(accum, bias):
    E = accum
    D = E + bias
    return D, E
"""
        example_tensors = {
            "accum": CutlassTensor(
                element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
            ),
            "bias": BIAS,
            # "beta": 0.5, TODO: mlazos support scalars
            # "alpha": 0.5, TODO: mlazos support scalars
            "D": CutlassTensor(
                element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
            ),
            "E": CutlassTensor(
                element=DataType.f32, shape=(M, N), layout_tag=LayoutType.RowMajor
            ),
        }

        cuda_arch = int(get_cuda_arch())  # type: ignore[arg-type]
        epilogue_functor = _trace(code, example_tensors, cuda_arch)

        self.assertExpectedInline(
            _render_argument_type(
                epilogue_functor,
                _create_mock_buffer_name_map(example_tensors),
                lambda x: int(x),
            ),
            """\
{ /* thread */
        { /* E */
          {}, /* accum */
          {/* ptr_aux */ (float*) E, /* dAux */ {2048, _1{}, _0{}}}, /* E */
        },
        {/* ptr_col */ (float*) bias, /* null_default */ float(0), /* dCol */ {}}, /* bias */
        {}, /* compute_0 */
      }
""",
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_evt_codegen(self):
        _, _, code = trace(
            BIAS_CODE,
            EXAMPLE_TENSORS,
            DataType.f32,
            DataType.f32,
            MockTileDescription(),
            EpilogueScheduleType.ScheduleAuto,
            _create_mock_buffer_name_map(EXAMPLE_TENSORS),
            lambda x: x,  # static shapes
        )
        self.assertExpectedInline(
            code,
            """\

using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
  cute::Shape<_128, _128, _8>, cutlass::epilogue::collective::EpilogueTileAuto,
  float, float,
  cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using ElementC = float;
using StrideC = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using TensorC = cutlass::epilogue::fusion::Sm90SrcFetch<float>;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

using AuxDescriptor = cutlass::epilogue::collective::detail::AuxLoadDescriptor<EpilogueDescriptor, \
cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, float>;

using Aux = cutlass::epilogue::fusion::Sm90AuxLoad<
    AuxDescriptor::Stages, typename AuxDescriptor::EpilogueTile, float,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, typename AuxDescriptor::SmemLayoutAtom, \
typename AuxDescriptor::CopyOpS2R
>;

using Bias = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, float, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<
    Compute0,
    Accum,
    TensorC>;

using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<
    Compute1,
    EVTCompute0,
    Aux>;

using FDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, float
>;

using F = cutlass::epilogue::fusion::Sm90AuxStore<
    FDescriptor::Stages, typename FDescriptor::EpilogueTile, float,
    cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::Int<1>, \
cute::Int<0>>, typename FDescriptor::SmemLayoutAtom,
    typename FDescriptor::CopyOpR2S
>;

using EVTF = cutlass::epilogue::fusion::Sm90EVT<
    F,
    EVTCompute1>;

using Compute2 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::epilogue::thread::ReLu, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute3 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute4 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using DagCompute4 = cutlass::epilogue::fusion::Sm90TopologicalVisitor<
    float,
    cute::tuple<
        cute::seq<>,
        cute::seq<>,
        cute::seq<0>,
        cute::seq<2, 1>,
        cute::seq<3, 0>,
    >,
    EVTF,
    Bias,
    Compute2,
    Compute3,
    Compute4
>;

using ElementD = float;
using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")

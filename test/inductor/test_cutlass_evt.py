# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._dynamo.test_case import TestCase
from torch._inductor.codegen.cuda.cutlass_utils import (
    torch_dtype_to_cutlass_type,
    try_import_cutlass,
)
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.virtualized import ops
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

    class MockNode:
        def __init__(self, name, shape, stride, dtype):
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.stride = stride

        def get_layout(self):
            class MockLayout:
                def __init__(self, shape, stride, dtype):
                    self.size = shape
                    self.stride = stride
                    self.dtype = dtype

            return MockLayout(self.shape, self.stride, self.dtype)

        def get_name(self):
            return self.name

    def _create_mock_buffer_name_map(example_tensors):
        name_to_buffer = {}
        for name, tensor in example_tensors.items():
            if isinstance(tensor, CutlassTensor):
                name_to_buffer[name] = MockNode(
                    name, tensor.shape, tensor.stride, torch.float32
                )

        return name_to_buffer


class MockData(Pointwise):
    @staticmethod
    def _index(a):  # typing: ignore
        return None


class MockIRNode(ComputedBuffer):
    def __init__(self, name, inner_fn):
        self.name = name
        self.data = MockData(device=None, dtype=None, inner_fn=inner_fn, ranges=None)

    def get_name(self):
        return self.name


class TestCutlassEVT(TestCase):
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_cutlass_py_codegen(self):
        from torch._inductor.codegen.cuda.cutlass_python_evt import CutlassEVTCodegen

        # buf0 is acc
        # buf1 is external
        def inner_fn_buf3(index):
            acc = ops.load("acc", index)
            buf1 = ops.load("buf1", index)
            buf2 = ops.load("buf2", index)
            return acc * buf1 + buf2

        def inner_fn_buf4(index):
            acc = ops.load("acc", index)
            buf3 = ops.load("buf3", index)
            return acc + buf3

        buf3_node = MockIRNode("buf3", inner_fn_buf3)
        buf4_node = MockIRNode("buf4", inner_fn_buf4)
        reads, writes, renames, code = CutlassEVTCodegen.ir_to_evt_python_code(
            "acc", [buf3_node, buf4_node]
        )
        self.assertExpectedInline(reads, """['acc', 'buf1', 'buf2']""")
        self.assertExpectedInline(writes, """['buf3', 'buf4']""")
        self.assertExpectedInline(
            renames, """{'buf3': 'D', 'buf4': 'tmp_2', 'acc': 'accum'}"""
        )
        self.assertExpectedInline(
            code,
            """\
def fn(accum, buf1, buf2):
    tmp_0 = accum * buf1
    tmp_1 = tmp_0 + buf2
    D = tmp_1
    tmp_2 = accum + D
    return D, tmp_2
""",
        )

    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_example_tensor_creation(self):
        from torch._inductor.codegen.cuda.cutlass_lib_extensions.evt_extensions import (
            create_example_tensors,
        )

        row_major_buf0 = MockNode("buf0", (3, 2, 1), (2, 1, 0), torch.float32)
        col_major_buf1 = MockNode("buf1", (3, 2, 1), (1, 3, 0), torch.float32)
        read_names = ["buf0"]
        write_names = ["buf1"]
        buffer_renames = {"buf0": "acc"}
        name_to_buffer = {"buf0": row_major_buf0, "buf1": col_major_buf1}
        result = create_example_tensors(
            read_names, write_names, buffer_renames, name_to_buffer
        )
        self.assertEqual(result["acc"].shape, (3, 2, 1))
        self.assertEqual(result["acc"].stride, (2, 1, 0))
        self.assertEqual(
            result["acc"].element, torch_dtype_to_cutlass_type(torch.float32)
        )

        self.assertEqual(result["buf1"].shape, (3, 2, 1))
        self.assertEqual(result["buf1"].stride, (1, 3, 0))
        self.assertEqual(
            result["buf1"].element, torch_dtype_to_cutlass_type(torch.float32)
        )

    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_evt_argument_codegen(self):
        epilogue_functor = _trace(BIAS_CODE, EXAMPLE_TENSORS)

        self.assertExpectedInline(
            _render_argument_type(
                epilogue_functor, _create_mock_buffer_name_map(EXAMPLE_TENSORS)
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

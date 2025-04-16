# Owner(s): ["module: inductor"]
import unittest

from torch._dynamo.test_case import TestCase
from torch._inductor.codegen.cuda.cutlass_utils import try_import_cutlass
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


if try_import_cutlass():
    import cutlass_library as cutlass_lib
    from cutlass_library import EpilogueScheduleType

    LayoutType = cutlass_lib.LayoutType
    DataType = cutlass_lib.DataType
    from torch._inductor.codegen.cuda.cutlass_lib_extensions.evt_extensions import (
        CutlassTensor,
        trace,
    )

    class MockTileDescription:
        threadblock_shape = (128, 128, 8)


class TestCutlassEVT(TestCase):
    @unittest.skipIf(not try_import_cutlass(), "requires cutlass")
    def test_evt_codegen(self):
        bias_code = """def example_epilogue(accum, alpha, C, beta, aux, bias):
    F = alpha * accum + (beta * C + aux)
    E = relu(F + 1) + bias
    D = E + F
    return D, F"""

        type_C = DataType.f32
        m = 4224
        n = 2048
        bias = CutlassTensor(
            shape=(m, 1), element=type_C, layout_tag=LayoutType.RowMajor
        )

        examples_tensors = {
            "accum": CutlassTensor(
                element=DataType.f32, shape=(m, n), layout_tag=LayoutType.RowMajor
            ),
            "bias": bias,
            "beta": 0.5,
            "alpha": 0.5,
            "D": CutlassTensor(
                element=DataType.f32, shape=(m, n), layout_tag=LayoutType.RowMajor
            ),
            "C": CutlassTensor(
                element=DataType.f32, shape=(m, n), layout_tag=LayoutType.RowMajor
            ),
            "F": CutlassTensor(
                element=DataType.f32, shape=(m, n), layout_tag=LayoutType.RowMajor
            ),
            "aux": CutlassTensor(
                element=DataType.f32, shape=(m, n), layout_tag=LayoutType.RowMajor
            ),
        }

        _, code = trace(
            bias_code,
            examples_tensors,
            DataType.f32,
            DataType.f32,
            MockTileDescription(),
            EpilogueScheduleType.ScheduleAuto,
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

using Alpha = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
    float, cute::Stride<cute::Int<0>, cute::Int<0>, cute::Int<0>>, 1, cutlass::multiplies
>;

using AuxDescriptor = cutlass::epilogue::collective::detail::AuxLoadDescriptor\
<EpilogueDescriptor, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, float>;

using Aux = cutlass::epilogue::fusion::Sm90AuxLoad<
    AuxDescriptor::Stages, typename AuxDescriptor::EpilogueTile, float,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, typename AuxDescriptor::SmemLayoutAtom, typename AuxDescriptor::CopyOpS2R
>;

using Beta = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
    float, cute::Stride<cute::Int<0>, cute::Int<0>, cute::Int<0>>, 1, cutlass::multiplies
>;

using Bias = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, float, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<
    Compute0,
    Alpha,
    Accum>;

using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<
    Compute1,
    Beta,
    TensorC>;

using Compute2 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute2 = cutlass::epilogue::fusion::Sm90EVT<
    Compute2,
    EVTCompute1,
    Aux>;

using Compute3 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute3 = cutlass::epilogue::fusion::Sm90EVT<
    Compute3,
    EVTCompute0,
    EVTCompute2>;

using FDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, float
>;

using F = cutlass::epilogue::fusion::Sm90AuxStore<
    FDescriptor::Stages, typename FDescriptor::EpilogueTile, float,
    cutlass::FloatRoundStyle::round_to_nearest, \
cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>, typename FDescriptor::SmemLayoutAtom,
    typename FDescriptor::CopyOpR2S
>;

using EVTF = cutlass::epilogue::fusion::Sm90EVT<
    F,
    EVTCompute3>;

using Imm10 = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
    float, cute::Stride<cute::Int<0>, cute::Int<0>, cute::Int<0>>, 1, cutlass::multiplies
>;

using Compute4 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute5 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::epilogue::thread::ReLu, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute6 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using Compute7 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using DagCompute7 = cutlass::epilogue::fusion::Sm90TopologicalVisitor<
    float,
    cute::tuple<
        cute::seq<>,
        cute::seq<>,
        cute::seq<>,
        cute::seq<0, 2>,
        cute::seq<3>,
        cute::seq<4, 1>,
        cute::seq<5, 0>,
    >,
    EVTF,
    Bias,
    Imm10,
    Compute4,
    Compute5,
    Compute6,
    Compute7
>;

using ElementD = float;
using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")

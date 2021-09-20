import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/test_tensorexpr"


class TestATen(TestCase):
    cpp_name = "ATen"

    def test__cast_Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_cast_Float")

    def test_negInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "negInt")

    def test_negFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "negFloat")

    def test_addInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "addInt")

    def test_addFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "addFloat")

    def test_subInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "subInt")

    def test_subFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "subFloat")

    def test_lerp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "lerp")

    def test_addcmulInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "addcmulInt")

    def test_addcmulFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "addcmulFloat")

    def test_mulInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "mulInt")

    def test_mulFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "mulFloat")

    def test_divInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "divInt")

    def test_divFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "divFloat")

    def test_maxInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "maxInt")

    def test_maxFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "maxFloat")

    def test_minInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "minInt")

    def test_minFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "minFloat")

    def test_reluInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "reluInt")

    def test_reluFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "reluFloat")

    def test_logFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "logFloat")

    def test_fastLogFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fastLogFloat")

    def test_fastTanhFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fastTanhFloat")

    def test_fastSigmoidFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fastSigmoidFloat")

    def test_log10Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "log10Float")

    def test_log2Float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "log2Float")

    def test_expFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "expFloat")

    def test_erfFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "erfFloat")

    def test_cosFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "cosFloat")

    def test_eqInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "eqInt")

    def test_geInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "geInt")

    def test_gtInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "gtInt")

    def test_leInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "leInt")

    def test_ltInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ltInt")


class TestBoundsInference(TestCase):
    cpp_name = "BoundsInference"

    def test__1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_1")

    def test__2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_2")

    def test__3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_3")

    def test__4(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_4")

    def test__5(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_5")

    def test__6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_6")

    def test_Adjacent(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Adjacent")

    def test_MultipleTopLoopLoad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultipleTopLoopLoad")

    def test_MultipleTopLoopStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultipleTopLoopStore")

    def test_CacheReads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CacheReads")

    def test_Flattened(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Flattened")

    def test_GetPotentialHazards(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetPotentialHazards")

    def test_GetPotentialHazardsLoopNoHazard(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetPotentialHazardsLoopNoHazard")

    def test_GetPotentialHazardsLoopCall(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetPotentialHazardsLoopCall")

    def test_GetPotentialHazardsLoopSplit(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetPotentialHazardsLoopSplit")

    def test_HasConflictingOverlapSameBufferWithPartialOverlap(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "HasConflictingOverlapSameBufferWithPartialOverlap",
        )

    def test_HasConflictingOverlapSameBufferWithFullOverlap(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HasConflictingOverlapSameBufferWithFullOverlap"
        )

    def test_HasConflictingOverlapSameBufferWithFullOverlapRAW(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "HasConflictingOverlapSameBufferWithFullOverlapRAW",
        )

    def test_HasConflictingOverlapSameBufferNotOverlapping(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HasConflictingOverlapSameBufferNotOverlapping"
        )

    def test_HasConflictingOverlap2DBufferWithOverlap(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HasConflictingOverlap2DBufferWithOverlap"
        )

    def test_HasConflictingOverlap2DBufferWithNoOverlap(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HasConflictingOverlap2DBufferWithNoOverlap"
        )

    def test_HasConflictingOverlapDifferentBuffers(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HasConflictingOverlapDifferentBuffers"
        )

    def test_HasConflictingOverlapDueToRAWDependence(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HasConflictingOverlapDueToRAWDependence"
        )

    def test_HasConflictingOverlapDueToWARDependence(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "HasConflictingOverlapDueToWARDependence"
        )

    def test_HasConflictingOverlapWithLoads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HasConflictingOverlapWithLoads")

    def test_IsOverlapping(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsOverlapping")


class TestConv(TestCase):
    cpp_name = "Conv"

    def test_Conv2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2D")


class TestCppPrinter(TestCase):
    cpp_name = "CppPrinter"

    def test_IntImm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IntImm")

    def test_FloatImm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatImm")

    def test_FloatImm1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatImm1")

    def test_DoubleImm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleImm")

    def test_DoubleImm1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleImm1")

    def test_HalfImm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HalfImm")

    def test_Add(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Add")

    def test_AddExpr1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AddExpr1")

    def test_AddExpr2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AddExpr2")

    def test_AddExpr3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AddExpr3")

    def test_Mod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Mod")

    def test_ModFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ModFloat")

    def test_Max(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Max")

    def test_MaxFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxFloat")

    def test_MaxHalf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxHalf")

    def test_And(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "And")

    def test_CompareSelect(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CompareSelect")

    def test_IfThenElse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IfThenElse")

    def test_AllocateFree(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllocateFree")

    def test_LoadStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoadStore")

    def test_Var(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Var")

    def test_Cast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cast")

    def test_BitCast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitCast")

    def test_Let(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Let")

    def test_For(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "For")

    def test_Cond(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cond")

    def test_Intrinsics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Intrinsics")

    def test_ExternalCall(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExternalCall")


class TestExpr(TestCase):
    cpp_name = "Expr"

    def test_BasicValueTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicValueTest")

    def test_BasicValueTest02(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicValueTest02")

    def test_LetTest01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LetTest01")

    def test_LetTest02(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LetTest02")

    def test_LetStmtTest01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LetStmtTest01")

    def test_IntTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IntTest")

    def test_FloatTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FloatTest")

    def test_ByteTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ByteTest")

    def test_CharTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CharTest")

    def test_ShortTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ShortTest")

    def test_LongTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LongTest")

    def test_HalfTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HalfTest")

    def test_DoubleTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleTest")

    def test_VectorAdd01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "VectorAdd01")

    def test_CompareSelectEQ(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CompareSelectEQ")

    def test_CompareSelectDtypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CompareSelectDtypes")

    def test_IntrinsicsDtypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IntrinsicsDtypes")

    def test_Substitute01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Substitute01")

    def test_Math01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Math01")

    def test_UnaryMath01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnaryMath01")

    def test_BinaryMath01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BinaryMath01")

    def test_LogicalOps01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LogicalOps01")

    def test_LogicalOps02(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LogicalOps02")

    def test_LogicalOps03(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LogicalOps03")

    def test_BitwiseOps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitwiseOps")

    def test_DynamicShapeAdd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DynamicShapeAdd")


class TestExternalCall(TestCase):
    cpp_name = "ExternalCall"

    def test_Conv2d_float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2d_float")

    def test_Conv2d_int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2d_int")

    def test_Conv2d_nobias_noargs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2d_nobias_noargs")

    def test_Addmm_float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Addmm_float")

    def test_Prepacked_Linear_float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Prepacked_Linear_float")

    def test_Prepacked_Conv2d_float(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Prepacked_Conv2d_float")

    def test_BinaryFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BinaryFloat")

    def test_UnaryFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnaryFloat")

    def test_ComputeInterop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComputeInterop")

    def test_Inlining(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inlining")


class TestGraphOpt(TestCase):
    cpp_name = "GraphOpt"

    def test_OptimizeCat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeCat")

    def test_OptimizeCat2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeCat2")

    def test_OptimizeCat3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeCat3")

    def test_OptimizeCatWithTypePromotionInUser(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeCatWithTypePromotionInUser")

    def test_OptimizeCatWithTypePromotionInCat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeCatWithTypePromotionInCat")

    def test_OptimizeCatNoSingleTensorElementwiseOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OptimizeCatNoSingleTensorElementwiseOp"
        )

    def test_OptimizeCatNoSingleTensorElementwiseOp2(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OptimizeCatNoSingleTensorElementwiseOp2"
        )


class TestIRPrinter(TestCase):
    cpp_name = "IRPrinter"

    def test_BasicValueTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicValueTest")

    def test_BasicValueTest02(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicValueTest02")

    def test_CastTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CastTest")

    def test_FunctionName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FunctionName")


class TestIRVerifier(TestCase):
    cpp_name = "IRVerifier"

    def test_BitwiseOps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitwiseOps")

    def test_CompareSelect(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CompareSelect")

    def test_Ramp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Ramp")

    def test_Load(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Load")

    def test_IfThenElse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IfThenElse")

    def test_For(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "For")

    def test_Block(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Block")

    def test_Store(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Store")


class TestKernel(TestCase):
    cpp_name = "Kernel"

    def test_InliningIntermediates(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InliningIntermediates")

    def test_PreAllocIntermediateBufs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PreAllocIntermediateBufs")

    def test__1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_1")

    def test__2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_2")

    def test__3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "_3")

    def test_Huge(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Huge")

    def test_ParallelStrided(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ParallelStrided")

    def test_DISABLED_Shape_Inference(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_Shape_Inference")

    def test_CatInputTypesPromotion(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CatInputTypesPromotion")

    def test_CatAndInlineWithAConstantDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CatAndInlineWithAConstantDim")

    def test_CatWoConditionals(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CatWoConditionals")

    def test_OptimizeConditionals(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeConditionals")

    def test_SumAllAxes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SumAllAxes")

    def test_SumOneAxis(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SumOneAxis")

    def test_SumMultipleAxes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SumMultipleAxes")

    def test_Softmax2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmax2D")

    def test_Softmax3D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmax3D")

    def test_Softmax4D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmax4D")

    def test_SignTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SignTest")

    def test_InlineProducerIntoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InlineProducerIntoReduction")

    def test_InlineReductionIntoConsumer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InlineReductionIntoConsumer")

    def test_ConstantTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantTensors")

    def test_ConstantTensorsNonContiguous(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantTensorsNonContiguous")

    def test_RunFast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RunFast")

    def test_CodegenInspection(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CodegenInspection")

    def test_CustomLowering(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CustomLowering")

    def test_Vectorize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Vectorize")

    def test_DISABLED_FlattenVectorize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_FlattenVectorize")


class TestLoopNest(TestCase):
    cpp_name = "LoopNest"

    def test_ExprSimple01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSimple01")

    def test_ExprLower01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprLower01")

    def test_ExprSimple02(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSimple02")

    def test_ExprSliceHeadWithLoopOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceHeadWithLoopOptions")

    def test_ExprSliceTailWithLoopOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceTailWithLoopOptions")

    def test_ExprSliceHeadWhenFactorEqualsSize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceHeadWhenFactorEqualsSize")

    def test_ExprSliceHeadWhenFactorLargerThanSize(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ExprSliceHeadWhenFactorLargerThanSize"
        )

    def test_ExprSliceHead(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceHead")

    def test_ExprSliceHeadWithNonZeroStart(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceHeadWithNonZeroStart")

    def test_ExprSliceTailWhenFactorEqualsSize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceTailWhenFactorEqualsSize")

    def test_ExprSliceTailWhenFactorLargerThanSize(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ExprSliceTailWhenFactorLargerThanSize"
        )

    def test_ExprSliceTail(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceTail")

    def test_ExprSplitAndSlice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSplitAndSlice")

    def test_ExprSliceAndNormalize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceAndNormalize")

    def test_ExprSliceWithVariableDimension(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSliceWithVariableDimension")

    def test_ExprSplitWithTail(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSplitWithTail")

    def test_ExprSplitWithTailNone(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSplitWithTailNone")

    def test_ExprSplitWithMask01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSplitWithMask01")

    def test_ExprSplitWithMaskRepeatedNoMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExprSplitWithMaskRepeatedNoMask")

    def test_getLoopAt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "getLoopAt")

    def test_TileSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TileSimple")

    def test_TileWithTails(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TileWithTails")

    def test_TileInMiddle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TileInMiddle")

    def test_SplitWithTailWithLoopOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SplitWithTailWithLoopOptions")

    def test_SplitWithMaskWithLoopOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SplitWithMaskWithLoopOptions")

    def test_ScheduleBroadcastAddBuffer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleBroadcastAddBuffer")

    def test_ScheduleFunctionCall01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleFunctionCall01")

    def test_ScheduleInlineSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineSimple")

    def test_ScheduleInlineFunc01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineFunc01")

    def test_ScheduleInlineRandom(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineRandom")

    def test_ScheduleInlineRandomUnrelated(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineRandomUnrelated")

    def test_ScheduleInlineRandomLowerDimensions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineRandomLowerDimensions")

    def test_ScheduleInlineIntrinsics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineIntrinsics")

    def test_ScheduleInlineRandWithIntrinsics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineRandWithIntrinsics")

    def test_ScheduleSplitAThenInline(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleSplitAThenInline")

    def test_ScheduleSplitBThenInline(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleSplitBThenInline")

    def test_ScheduleSplitTwiceThenInline(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleSplitTwiceThenInline")

    def test_ScheduleInlineThenSplit(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineThenSplit")

    def test_ScheduleSplitInlineThenSplit(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleSplitInlineThenSplit")

    def test_ScheduleSplitInlineSimplify(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleSplitInlineSimplify")

    def test_ScheduleInlineThreeMixedOnce(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineThreeMixedOnce")

    def test_ScheduleInlineThreeMixedTwice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineThreeMixedTwice")

    def test_ScheduleInlineThreeMixedInner(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineThreeMixedInner")

    def test_ScheduleInlineThreeMixedSplit(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineThreeMixedSplit")

    def test_ScheduleInlineOutputTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineOutputTensors")

    def test_ScheduleInlineWithCompoundIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleInlineWithCompoundIndices")

    def test_ScheduleInlineConsumerIndicesWithCast(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ScheduleInlineConsumerIndicesWithCast"
        )

    def test_ScheduleInlineProducerIndicesWithCast(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ScheduleInlineProducerIndicesWithCast"
        )

    def test_ScheduleFuserStyle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleFuserStyle")

    def test_ScheduleFuserThreeArg(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleFuserThreeArg")

    def test_ScheduleDynamicShape2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScheduleDynamicShape2D")

    def test_LoopNestComputeAt_1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestComputeAt_1")

    def test_LoopNestComputeAt_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestComputeAt_2")

    def test_LoopNestComputeAt_3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestComputeAt_3")

    def test_Reduce2dComputeAt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reduce2dComputeAt")

    def test_DISABLED_Conv1d_NH(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_Conv1d_NH")

    def test_LoopNestReorderAxis1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestReorderAxis1")

    def test_LoopNestReorderPartialAxes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestReorderPartialAxes")

    def test_LoopNestReorderInternalAxis(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestReorderInternalAxis")

    def test_LoopNestReorderEnclosingAxis(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestReorderEnclosingAxis")

    def test_LoopNestReorderSameAxis(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestReorderSameAxis")

    def test_LoopNestReorderExtraStatements(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestReorderExtraStatements")

    def test_LoopNestReorderLongStringOfPreOrphans(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "LoopNestReorderLongStringOfPreOrphans"
        )

    def test_LoopNestReorderLongStringOfPostOrphans(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "LoopNestReorderLongStringOfPostOrphans"
        )

    def test_LoopNestReorderLongStringFull(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestReorderLongStringFull")

    def test_LoopNestReorderInternalLoopNest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopNestReorderInternalLoopNest")

    def test_OuterLoopVectorization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OuterLoopVectorization")

    def test_VectorizeLoopNotNormalized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "VectorizeLoopNotNormalized")

    def test_Unroll(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Unroll")

    def test_UnrollOuter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnrollOuter")

    def test_UnrollInner(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnrollInner")

    def test_UnrollMultipleStatements(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnrollMultipleStatements")

    def test_UnrollNonLiteralConstantBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnrollNonLiteralConstantBounds")

    def test_UnrollEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnrollEmpty")

    def test_NoUnroll(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoUnroll")

    def test_UnrollWithLet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnrollWithLet")

    def test_IsNormalized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsNormalized")

    def test_NormalizeStartPositive(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NormalizeStartPositive")

    def test_NormalizeStartNegative(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NormalizeStartNegative")

    def test_NormalizeStartZero(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NormalizeStartZero")

    def test_NormalizeStartVariable(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NormalizeStartVariable")

    def test_NormalizeOnNestedOuterLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NormalizeOnNestedOuterLoop")

    def test_NormalizeOnNestedInnerLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NormalizeOnNestedInnerLoop")

    def test_NormalizeAndSplitWithTail(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NormalizeAndSplitWithTail")

    def test_FlattenSimpleLoopNest2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FlattenSimpleLoopNest2D")

    def test_FlattenSimpleLoopNest3D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FlattenSimpleLoopNest3D")

    def test_FlattenLoopNestAfterNormalize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FlattenLoopNestAfterNormalize")

    def test_FlattenLoopNestWithNonLiteralConstantBounds(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "FlattenLoopNestWithNonLiteralConstantBounds"
        )

    def test_FlattenImperfectLoopNest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FlattenImperfectLoopNest")

    def test_FlattenReductionLoopNest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FlattenReductionLoopNest")

    def test_FlattenReductionLoopNestFromTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FlattenReductionLoopNestFromTensor")

    def test_FlattenIncorrectLoopsAsInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FlattenIncorrectLoopsAsInput")

    def test_DetectInlineRankMismatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DetectInlineRankMismatch")

    def test_CacheReadsSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CacheReadsSimple")

    def test_CacheReadsOuter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CacheReadsOuter")

    def test_CacheReadsInternal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CacheReadsInternal")

    def test_CacheReadsInner(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CacheReadsInner")

    def test_CacheWritesSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CacheWritesSimple")

    def test_DeadStoreElimination(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeadStoreElimination")

    def test_DeadStoreEliminationWithIntermediates(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DeadStoreEliminationWithIntermediates"
        )

    def test_CompoundTensorSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CompoundTensorSimple")

    def test_InlineConstantIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InlineConstantIndex")

    def test_CompoundTensorUsed(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CompoundTensorUsed")

    def test_InlineFromLoad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InlineFromLoad")

    def test_OptimizeConditionalsSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeConditionalsSimple")

    def test_OptimizeConditionalsNestedConditions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeConditionalsNestedConditions")

    def test_OptimizeConditionalsMultipleStores(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeConditionalsMultipleStores")

    def test_OptimizeConditionalsMultipleStoresInOneLoop(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OptimizeConditionalsMultipleStoresInOneLoop"
        )

    def test_OptimizeConditionalsOuterLoopVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeConditionalsOuterLoopVar")

    def test_OptimizeConditionalsCompValuesNotOrdered(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OptimizeConditionalsCompValuesNotOrdered"
        )

    def test_OptimizeConditionalsCompValuesNotConstants(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OptimizeConditionalsCompValuesNotConstants"
        )

    def test_OptimizeConditionalsInvalidCondition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeConditionalsInvalidCondition")

    def test_OptimizeConditionalsInvalidCondition2(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OptimizeConditionalsInvalidCondition2"
        )

    def test_OptimizeConditionalsInvalidCondition3(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OptimizeConditionalsInvalidCondition3"
        )

    def test_OptimizeConditionalsInvalidCondition4(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OptimizeConditionalsInvalidCondition4"
        )

    def test_OptimizeConditionalsNotNormalized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizeConditionalsNotNormalized")

    def test_ColReduceSplitTailEvenReorder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ColReduceSplitTailEvenReorder")

    def test_ColReduceSplitTailUnevenReorder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ColReduceSplitTailUnevenReorder")

    def test_ColReduceSplitMaskEvenReorder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ColReduceSplitMaskEvenReorder")

    def test_ColReduceSplitMaskUnevenReorder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ColReduceSplitMaskUnevenReorder")

    def test_ReorderAxisWithMultipleConds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReorderAxisWithMultipleConds")

    def test_VectorizeUse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "VectorizeUse")

    def test_Int64Direct(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Int64Direct")

    def test_Int64Compute(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Int64Compute")

    def test_DistributeLoopWithAllStmtsAsPivots(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DistributeLoopWithAllStmtsAsPivots")

    def test_DistributeLoopWithOneStmtAsPivot(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DistributeLoopWithOneStmtAsPivot")

    def test_DistributeLoopWithoutAnyPivot(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DistributeLoopWithoutAnyPivot")

    def test_DistributeLoopOverInnerLoops(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DistributeLoopOverInnerLoops")

    def test_DistributeLoopAndParentsWithoutAnyPivot(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DistributeLoopAndParentsWithoutAnyPivot"
        )

    def test_fuseLoopsSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsSimple")

    def test_fuseLoopsMultiple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsMultiple")

    def test_fuseLoopsNested(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsNested")

    def test_fuseLoopsNested2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsNested2D")

    def test_fuseLoopsNested2DInner(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsNested2DInner")

    def test_fuseLoopsDifferentStopBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsDifferentStopBounds")

    def test_fuseLoopsDifferentStartBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsDifferentStartBounds")

    def test_fuseLoopsNotContiguous(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsNotContiguous")

    def test_fuseLoopsWithDifferentParents(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWithDifferentParents")

    def test_fuseLoopsWithVariableBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWithVariableBounds")

    def test_fuseLoopsWithExprBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWithExprBounds")

    def test_fuseLoopsWithDifferentExprBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWithDifferentExprBounds")

    def test_fuseLoopsWithNonOverlappingBufferAccesses(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "fuseLoopsWithNonOverlappingBufferAccesses"
        )

    def test_fuseLoopsWithNonOverlapping2DBufferAccesses(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "fuseLoopsWithNonOverlapping2DBufferAccesses"
        )

    def test_fuseLoopsWithReductions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWithReductions")

    def test_fuseLoopsWith2DReductions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWith2DReductions")

    def test_fuseLoopsWithComplexIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWithComplexIndices")

    def test_fuseLoopsWithMixedLoopVarsAsIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWithMixedLoopVarsAsIndices")

    def test_fuseLoopsWithTranspose(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsWithTranspose")

    def test_fuseLoopsThatViolateDependencies1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsThatViolateDependencies1")

    def test_fuseLoopsThatViolateDependencies2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsThatViolateDependencies2")

    def test_fuseLoopsThatViolateDependencies3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsThatViolateDependencies3")

    def test_fuseLoopsThatViolateDependencies4(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsThatViolateDependencies4")

    def test_fuseLoopsThatViolateDependencies5(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsThatViolateDependencies5")

    def test_fuseLoopsThatViolateDependencies6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsThatViolateDependencies6")

    def test_fuseLoopsThatViolateDependencies7(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fuseLoopsThatViolateDependencies7")

    def test_areLoopsPerfectlyNested(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "areLoopsPerfectlyNested")

    def test_reorderNestedLoops2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "reorderNestedLoops2D")

    def test_reorderNestedLoops3D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "reorderNestedLoops3D")

    def test_reorderNestedLoops4D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "reorderNestedLoops4D")

    def test_reorderTrivialPermutation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "reorderTrivialPermutation")

    def test_reorderInvalidPermutations(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "reorderInvalidPermutations")

    def test_reorderInvalidLoopNest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "reorderInvalidLoopNest")

    def test_compressBufferSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "compressBufferSimple")

    def test_compressBufferMultipleDims(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "compressBufferMultipleDims")

    def test_compressBufferMultipleDims2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "compressBufferMultipleDims2")

    def test_compressBufferDifferentOrderIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "compressBufferDifferentOrderIndices")

    def test_compressBufferVariableBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "compressBufferVariableBounds")

    def test_compressBufferNoCommonParentLoops(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "compressBufferNoCommonParentLoops")

    def test_compressBufferIndicesMixed(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "compressBufferIndicesMixed")

    def test_compressMultipleBuffers(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "compressMultipleBuffers")

    def test_sanitizeNames(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "sanitizeNames")


class TestMemDependency(TestCase):
    cpp_name = "MemDependency"

    def test_BoundOverlap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BoundOverlap")

    def test_BoundOverlapSymbolic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BoundOverlapSymbolic")

    def test_BoundOverlapMultiDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BoundOverlapMultiDim")

    def test_BoundSubtract(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BoundSubtract")

    def test_BoundSubtractSymbolic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BoundSubtractSymbolic")

    def test_BoundSubtractMultiDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BoundSubtractMultiDim")

    def test_BoundSubtractMultiDimSymbolic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BoundSubtractMultiDimSymbolic")

    def test_MemDependencyCheckerSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerSimple")

    def test_MemDependencyCheckerMultiStmt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerMultiStmt")

    def test_MemDependencyCheckerOverlap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerOverlap")

    def test_MemDependencyCheckerLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerLoop")

    def test_MemDependencyCheckerLoopReduce(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerLoopReduce")

    def test_MemDependencyCheckerLoopReduceExpanded(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MemDependencyCheckerLoopReduceExpanded"
        )

    def test_MemDependencyCheckerInputsOutputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerInputsOutputs")

    def test_MemDependencyCheckerOutputDoesntDepend(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MemDependencyCheckerOutputDoesntDepend"
        )

    def test_MemDependencyCheckerLoopBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerLoopBounds")

    def test_MemDependencyCheckerLoopBoundsIndexShift(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MemDependencyCheckerLoopBoundsIndexShift"
        )

    def test_MemDependencyCheckerLoopSelfDependency(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MemDependencyCheckerLoopSelfDependency"
        )

    def test_MemDependencyCheckerLoopDistinctStrides(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MemDependencyCheckerLoopDistinctStrides"
        )

    def test_MemDependencyCheckerLoopBoundsCond(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerLoopBoundsCond")

    def test_MemDependencyCheckerIfThenElse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerIfThenElse")

    def test_MemDependencyCheckerCutLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerCutLoop")

    def test_MemDependencyCheckerDynamicShapes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerDynamicShapes")

    def test_MemDependencyCheckerMultiDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerMultiDim")

    def test_MemDependencyCheckerComputeAPI(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerComputeAPI")

    def test_MemDependencyCheckerComputeInline(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerComputeInline")

    def test_MemDependencyCheckerComputeSplit(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerComputeSplit")

    def test_MemDependencyCheckerComputeReorder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerComputeReorder")

    def test_MemDependencyCheckerComputeReduce(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerComputeReduce")

    def test_MemDependencyCheckerComputeGEMM(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MemDependencyCheckerComputeGEMM")


class TestReductions(TestCase):
    cpp_name = "Reductions"

    def test_ReduceSum0D_1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSum0D_1")

    def test_ReduceSum0D_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSum0D_2")

    def test_ReduceSum1D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSum1D")

    def test_ReduceSum2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSum2D")

    def test_ReduceSum3D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSum3D")

    def test_ReduceSum10D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSum10D")

    def test_ReduceProduct(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceProduct")

    def test_ReduceMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceMax")

    def test_ReduceMinCustomInitializer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceMinCustomInitializer")

    def test_ReduceAnyAll(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceAnyAll")

    def test_ReduceMatmul2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceMatmul2D")

    def test_ReduceRfactorLike(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceRfactorLike")

    def test_ReduceAsProducer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceAsProducer")

    def test_ReduceAsConsumer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceAsConsumer")

    def test_SplitReduceAxis(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SplitReduceAxis")

    def test_SplitNonReduceAxis(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SplitNonReduceAxis")

    def test_ReorderedReductionInitializer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReorderedReductionInitializer")

    def test_ReduceRfactor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceRfactor")

    def test_Reduce3DRfactorInner(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reduce3DRfactorInner")

    def test_Reduce3DRfactorOuter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reduce3DRfactorOuter")

    def test_ReduceRepeatedInternalRfactor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceRepeatedInternalRfactor")

    def test_ReduceSplitTail(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSplitTail")

    def test_ReduceSplitNoTail(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSplitNoTail")

    def test_ReduceOverSplitTail(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceOverSplitTail")

    def test_ReduceSplitMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSplitMask")

    def test_ReduceSplitNoMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSplitNoMask")

    def test_ReduceOverSplitMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceOverSplitMask")

    def test_ReduceSplitRfactor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceSplitRfactor")

    def test_ReduceOverSplitRfactor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceOverSplitRfactor")

    def test_ReduceInlineReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceInlineReduction")

    def test_ReduceInlineConsumer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceInlineConsumer")

    def test_ReduceInlineReducerInternal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReduceInlineReducerInternal")

    def test_ReductionCacheAccessesOperatorAxis(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionCacheAccessesOperatorAxis")

    def test_ReductionCacheAccessesOuterReduceAxis(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ReductionCacheAccessesOuterReduceAxis"
        )

    def test_ReductionCacheAccessesInnerReduceAxis(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ReductionCacheAccessesInnerReduceAxis"
        )

    def test_ReductionCacheBodyAccess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionCacheBodyAccess")

    def test_ReductionCacheConsumerAccess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionCacheConsumerAccess")

    def test_ReductionSplitCacheConsumerAccess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionSplitCacheConsumerAccess")

    def test_ReductionReorderCacheConsumerAccess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionReorderCacheConsumerAccess")

    def test_ReductionRfactorCacheTempOuter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionRfactorCacheTempOuter")

    def test_ReductionRfactorCacheTempInner(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionRfactorCacheTempInner")

    def test_ReductionVectorize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionVectorize")

    def test_ReductionVectorizeInner(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionVectorizeInner")

    def test_ReductionVectorizeRfactor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReductionVectorizeRfactor")

    def test_InitFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InitFunction")


class TestRegisterizer(TestCase):
    cpp_name = "Registerizer"

    def test_RegisterizerSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerSimple")

    def test_RegisterizerLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerLoop")

    def test_RegisterizerLoopFixedLoad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerLoopFixedLoad")

    def test_RegisterizerLoopInternal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerLoopInternal")

    def test_RegisterizerLoopInternalLoadOverlap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerLoopInternalLoadOverlap")

    def test_RegisterizerLoopInternalRepeated(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerLoopInternalRepeated")

    def test_RegisterizerLoopInternalRepeatedOverlapLoopVar(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterizerLoopInternalRepeatedOverlapLoopVar"
        )

    def test_RegisterizerLoopInternalRepeatedOverlapOther(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterizerLoopInternalRepeatedOverlapOther"
        )

    def test_RegisterizerMultiVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiVar")

    def test_RegisterizerVariableLoad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerVariableLoad")

    def test_RegisterizerSymbolicIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerSymbolicIndices")

    def test_RegisterizerMultiLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiLoop")

    def test_RegisterizerRepeated(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerRepeated")

    def test_RegisterizerNoLoads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNoLoads")

    def test_RegisterizerNoRepeatedStores(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNoRepeatedStores")

    def test_RegisterizerMultiVarOverlap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiVarOverlap")

    def test_RegisterizerAllocs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerAllocs")

    def test_RegisterizerNoInitializer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNoInitializer")

    def test_RegisterizerNoInitializerLoopVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNoInitializerLoopVar")

    def test_RegisterizerLoadThenStore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerLoadThenStore")

    def test_RegisterizerParallelized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerParallelized")

    def test_RegisterizerConditionAfter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerConditionAfter")

    def test_RegisterizerConditionBefore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerConditionBefore")

    def test_RegisterizerConditionInside(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerConditionInside")

    def test_RegisterizerConditionInsideOverlap1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerConditionInsideOverlap1")

    def test_RegisterizerConditionInsideOverlap2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerConditionInsideOverlap2")

    def test_RegisterizerConditionHidden(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerConditionHidden")

    def test_RegisterizerConditionUnhidden(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerConditionUnhidden")

    def test_RegisterizerCondCondition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerCondCondition")

    def test_RegisterizerCondConditionUnhidden(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerCondConditionUnhidden")

    def test_RegisterizerIfThenElseHidden(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerIfThenElseHidden")

    def test_RegisterizerIfThenElseUnhidden(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerIfThenElseUnhidden")

    def test_RegisterizerIfThenElseNested(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerIfThenElseNested")

    def test_RegisterizerIfThenElseInternal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerIfThenElseInternal")

    def test_RegisterizerIfThenElseCondition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerIfThenElseCondition")

    def test_RegisterizerIfThenElseConditionUnhidden(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterizerIfThenElseConditionUnhidden"
        )

    def test_RegisterizerConditionBranchOnly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerConditionBranchOnly")

    def test_RegisterizerCondIfThenElse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerCondIfThenElse")

    def test_RegisterizerIfThenElseLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerIfThenElseLoop")

    def test_RegisterizerIfThenElseLoopCut(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerIfThenElseLoopCut")

    def test_RegisterizerPartialAfter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerPartialAfter")

    def test_RegisterizerPartialBefore(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerPartialBefore")

    def test_RegisterizerPartialInside(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerPartialInside")

    def test_RegisterizerPartialCondition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerPartialCondition")

    def test_RegisterizerPartialConditionInternalCut(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterizerPartialConditionInternalCut"
        )

    def test_RegisterizerPartialConditionInternalStart(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterizerPartialConditionInternalStart"
        )

    def test_RegisterizerPartialOverlapsTwo(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerPartialOverlapsTwo")

    def test_RegisterizerNestedBlocks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNestedBlocks")

    def test_RegisterizerNestedConditions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNestedConditions")

    def test_RegisterizerNestedConditionsUnhidden(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNestedConditionsUnhidden")

    def test_RegisterizerNestedConditionsHiddenFirst(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterizerNestedConditionsHiddenFirst"
        )

    def test_RegisterizerNestedConditionsHiddenSecond(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterizerNestedConditionsHiddenSecond"
        )

    def test_RegisterizerNestedConditionsCut(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNestedConditionsCut")

    def test_RegisterizerNestedConditionLoopHidden(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterizerNestedConditionLoopHidden"
        )

    def test_RegisterizerNestedConditionThreeDeep(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNestedConditionThreeDeep")

    def test_RegisterizerNestedLoopSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerNestedLoopSimple")

    def test_RegisterizerHiddenAccessYes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerHiddenAccessYes")

    def test_RegisterizerHiddenAccessNo(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerHiddenAccessNo")

    def test_RegisterizerHiddenAccessMultiLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerHiddenAccessMultiLoop")

    def test_RegisterizerTwoConditionalLoops(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerTwoConditionalLoops")

    def test_RegisterizerTwoConditionalLoopsCut(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerTwoConditionalLoopsCut")

    def test_RegisterizerLoopLetVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerLoopLetVar")

    def test_RegisterizerLoopLetVarOuter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerLoopLetVarOuter")

    def test_RegisterizerMultiDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiDim")

    def test_RegisterizerMultiDimPartial(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiDimPartial")

    def test_RegisterizerMultiDimOverlap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiDimOverlap")

    def test_RegisterizerMultiDimPartialOverlap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiDimPartialOverlap")

    def test_RegisterizerMultiDim3DReduction1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiDim3DReduction1")

    def test_RegisterizerMultiDim3DReduction2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterizerMultiDim3DReduction2")


class TestSimplify(TestCase):
    cpp_name = "Simplify"

    def test_ConstantFoldSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldSimple")

    def test_ConstantFoldTwoLayer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldTwoLayer")

    def test_ConstantFoldShifts(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldShifts")

    def test_ConstantFoldBitwise(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldBitwise")

    def test_ConstantFoldMultiOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldMultiOp")

    def test_ConstantFoldMinMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldMinMax")

    def test_ConstantFoldIntrinsics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldIntrinsics")

    def test_ConstantFoldCastToBool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldCastToBool")

    def test_ConstantFoldWithVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantFoldWithVar")

    def test_ConditionalSelectFoldSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConditionalSelectFoldSimple")

    def test_ConditionalSelectFoldTwoLayer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConditionalSelectFoldTwoLayer")

    def test_ConditionalSelectFoldWithVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConditionalSelectFoldWithVar")

    def test_UnFoldableExpr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnFoldableExpr")

    def test_HashSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashSimple")

    def test_HashEquivalence(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashEquivalence")

    def test_HashEquivalenceRand(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashEquivalenceRand")

    def test_HashEquivalenceAfterFolding(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashEquivalenceAfterFolding")

    def test_HashDifferenceTypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashDifferenceTypes")

    def test_HashLargeExpression(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashLargeExpression")

    def test_HashForLoopOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HashForLoopOptions")

    def test_SimplifyAdd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyAdd")

    def test_SimplifySub(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifySub")

    def test_SimplifyMultiLayer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyMultiLayer")

    def test_SimplifyMultiTerm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyMultiTerm")

    def test_SimplifyCasts(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyCasts")

    def test_SimplifyEliminatesNoOps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyEliminatesNoOps")

    def test_SimplifyMultiVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyMultiVar")

    def test_DISABLED_SimplifyReorderings(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_SimplifyReorderings")

    def test_SimplifyEliminatesVar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyEliminatesVar")

    def test_SimplifyAdds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyAdds")

    def test_SimplifyMuls(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyMuls")

    def test_SimplifySubs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifySubs")

    def test_SimplifyDiv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDiv")

    def test_SimplifyDivWithLoopContext0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivWithLoopContext0")

    def test_SimplifyDivWithLoopContext1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivWithLoopContext1")

    def test_SimplifyDivWithLoopContext2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivWithLoopContext2")

    def test_SimplifyDivWithLoopContext3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivWithLoopContext3")

    def test_SimplifyDivWithLoopContext4(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivWithLoopContext4")

    def test_SimplifyDivWithLoopContext5(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivWithLoopContext5")

    def test_SimplifyDivWithLoopContext6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivWithLoopContext6")

    def test_SimplifyDivWithLoopContext7(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivWithLoopContext7")

    def test_SimplifyModWithLoopContext0(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModWithLoopContext0")

    def test_SimplifyModWithLoopContext1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModWithLoopContext1")

    def test_SimplifyModWithLoopContext2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModWithLoopContext2")

    def test_SimplifyModWithLoopContext3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModWithLoopContext3")

    def test_SimplifyModWithLoopContext4(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModWithLoopContext4")

    def test_SimplifyModWithLoopContext5(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModWithLoopContext5")

    def test_SimplifyModWithLoopContext6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModWithLoopContext6")

    def test_SimplifyModWithLoopContext7(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModWithLoopContext7")

    def test_SimplifyMod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyMod")

    def test_SimplifyMultiOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyMultiOp")

    def test_SimplifyManyOps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyManyOps")

    def test_SimplifyFactorization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyFactorization")

    def test_SimplifyFactorizeUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyFactorizeUneven")

    def test_SimplifyDeeperTerms(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDeeperTerms")

    def test_SimplifyDeeperDifference(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDeeperDifference")

    def test_SimplifyFoldComplexDifference(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyFoldComplexDifference")

    def test_SimplifyIfComponents(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyIfComponents")

    def test_SimplifyOpaqueTerms(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyOpaqueTerms")

    def test_SimplifySymbolicMinMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifySymbolicMinMax")

    def test_SimplifyNestedMax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyNestedMax")

    def test_SimplifyNestedMin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyNestedMin")

    def test_SimplifyWontReorderFloat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyWontReorderFloat")

    def test_SimplifyRoundModPattern(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyRoundModPattern")

    def test_SimplifyRoundModPatternFactorization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyRoundModPatternFactorization")

    def test_SimplifyRoundModPatternMultivar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyRoundModPatternMultivar")

    def test_SimplifyModRoundModPattern(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModRoundModPattern")

    def test_SimplifyModRoundModPatternFactorization(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "SimplifyModRoundModPatternFactorization"
        )

    def test_SimplifyModRoundModPatternMultivar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyModRoundModPatternMultivar")

    def test_SimplifyDivisionScalarFactorization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyDivisionScalarFactorization")

    def test_SimplifyConstantBranches(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyConstantBranches")

    def test_SimplifyConstantCond(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyConstantCond")

    def test_SimplifyEliminateEmptyCond(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyEliminateEmptyCond")

    def test_SimplifyConstantComparisons(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyConstantComparisons")

    def test_SimplifySymbolicComparisons(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifySymbolicComparisons")

    def test_SimplifyEliminateZeroLengthFor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyEliminateZeroLengthFor")

    def test_SimplifyOneLoopFor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyOneLoopFor")

    def test_SimplifyForWontLoseLoopOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyForWontLoseLoopOptions")

    def test_SimplifyMultilevelFor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyMultilevelFor")

    def test_SimplifyForCleansUp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyForCleansUp")

    def test_SimplifyEliminateEmptyFor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyEliminateEmptyFor")

    def test_SimplifyFlattenBlock(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyFlattenBlock")

    def test_SimplifyEliminateZeroLengthAlloc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyEliminateZeroLengthAlloc")

    def test_DontSimplifyRand(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DontSimplifyRand")

    def test_SimplifyReorderForCond(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyReorderForCond")

    def test_SimplifyFuseConditions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyFuseConditions")

    def test_SimplifySyncThreads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifySyncThreads")

    def test_SimplifyRampSubBroadcast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyRampSubBroadcast")

    def test_SimplifyBroadcastTermExpander(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimplifyBroadcastTermExpander")

    def test_DISABLED_CompareSelectCondAlwaysInLoopBounds(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DISABLED_CompareSelectCondAlwaysInLoopBounds"
        )

    def test_DISABLED_IfThenCondAlwaysInLoopBounds(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DISABLED_IfThenCondAlwaysInLoopBounds"
        )

    def test_DISABLED_MultiClauseCondAlwaysInLoopBounds(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DISABLED_MultiClauseCondAlwaysInLoopBounds"
        )

    def test_DISABLED_SimplifyLoopBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_SimplifyLoopBounds")


class TestTEFuserPass(TestCase):
    cpp_name = "TEFuserPass"

    def test_FuserPass_1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_1")

    def test_FuserPass_2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_2")

    def test_FuserPass_3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_3")

    def test_FuserPass_0DimInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_0DimInput")

    def test_FuserPass_UnfusibleDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_UnfusibleDevice")

    def test_FuserPass_UnknownShapes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_UnknownShapes")

    def test_FuserPass_Multidevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_Multidevice")

    def test_FuserPass_MergeGroups(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_MergeGroups")

    def test_FuserPass_UnknownShapesIgnored(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_UnknownShapesIgnored")

    def test_FuserPass_IgnoreUnknownShapeAtStart(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_IgnoreUnknownShapeAtStart")

    def test_FuserPass_Where(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_Where")

    def test_FuserPass_WhereList(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FuserPass_WhereList")


class TestType(TestCase):
    cpp_name = "Type"

    def test_Test01(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Test01")

    def test_BitCasting(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BitCasting")

    def test_Propagation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Propagation")


if __name__ == "__main__":
    run_tests()

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/test_jit"


class TestTopologicalMoveTest(TestCase):
    cpp_name = "TopologicalMoveTest"

    def test_SplitsDeps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SplitsDeps")

    def test_MoveAfterBackwardSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAfterBackwardSimple")

    def test_MoveAfterBackwardInvalid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAfterBackwardInvalid")

    def test_MoveAfterNoOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAfterNoOp")

    def test_MoveAfterBackwardMultipleDeps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAfterBackwardMultipleDeps")

    def test_MoveAfterBackwardNonZeroWorkingSet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAfterBackwardNonZeroWorkingSet")

    def test_MoveAfterForwardSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAfterForwardSimple")

    def test_MoveAfterForwardNonZeroWorkingSet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAfterForwardNonZeroWorkingSet")

    def test_MoveBeforeForwardSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveBeforeForwardSimple")

    def test_MoveBeforeBackwardSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveBeforeBackwardSimple")

    def test_MoveBeforeNoOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveBeforeNoOp")

    def test_MoveBeforeForwardWithDeps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveBeforeForwardWithDeps")

    def test_MoveBeforeBackwardWithDeps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveBeforeBackwardWithDeps")

    def test_DepsDisallowMove(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DepsDisallowMove")

    def test_MoveAfterBeforeWithDeps(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAfterBeforeWithDeps")


class TestAliasAnalysisTest(TestCase):
    cpp_name = "AliasAnalysisTest"

    def test_AliasingMutationBlocksMoves(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AliasingMutationBlocksMoves")

    def test_AliasingMutationBlocksMoves2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AliasingMutationBlocksMoves2")

    def test_SideEffectsBlockMoves(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SideEffectsBlockMoves")

    def test_MovingAcrossInnerBlocks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MovingAcrossInnerBlocks")

    def test_NoneHasNoWriters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoneHasNoWriters")

    def test_SafeToChangeAliasingRelationship(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SafeToChangeAliasingRelationship")


class TestWriteTrackingTest(TestCase):
    cpp_name = "WriteTrackingTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_IsMutable(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsMutable")

    def test_IsImmutable(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsImmutable")

    def test_HasWriters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HasWriters")


class TestContainerAliasingTest(TestCase):
    cpp_name = "ContainerAliasingTest"

    def test_MayContainAlias(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MayContainAlias")

    def test_MayContainAlias_cast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MayContainAlias_cast")

    def test_PrimitveValuesDontAliasContainers(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrimitveValuesDontAliasContainers")

    def test_UnionAliasing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnionAliasing")

    def test_InputsCanAliasOutputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InputsCanAliasOutputs")

    def test_NestedTupleConstruct(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedTupleConstruct")

    def test_NestedTypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedTypes")

    def test_Simple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Simple")

    def test_Lists(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Lists")

    def test_Lists2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Lists2")

    def test_Conservative(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conservative")

    def test_MovesAcrossContainedWrites(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MovesAcrossContainedWrites")

    def test_MovesAcrossContainedWritesNested(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MovesAcrossContainedWritesNested")


class TestWildcardsTest(TestCase):
    cpp_name = "WildcardsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_TypeIsolation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeIsolation")

    def test_InvariantContainerAliasing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InvariantContainerAliasing")


class TestAliasRegistrationTest(TestCase):
    cpp_name = "AliasRegistrationTest"

    def test_ConservativeWithInferredSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConservativeWithInferredSchema")

    def test_ConservativeWithSpecifiedSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConservativeWithSpecifiedSchema")

    def test_ConservativeWithAliasingAnnotationsShouldError(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ConservativeWithAliasingAnnotationsShouldError"
        )

    def test_ConservativeWithAliasingAnnotationsShouldError2(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ConservativeWithAliasingAnnotationsShouldError2",
        )

    def test_FromSchemaWithInferredSchemaShouldError(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "FromSchemaWithInferredSchemaShouldError"
        )

    def test_FromSchemaInferredPure(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FromSchemaInferredPure")

    def test_FromSchemaAliased(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FromSchemaAliased")

    def test_FromSchemaPure(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FromSchemaPure")

    def test_PureNoSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PureNoSchema")

    def test_PureWithSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PureWithSchema")

    def test_PureWithAnnotationsShouldError(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PureWithAnnotationsShouldError")

    def test_AliasMoveAtenListOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AliasMoveAtenListOp")

    def test_PureWithAnnotationsShouldError2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PureWithAnnotationsShouldError2")


class TestAutodiffTest(TestCase):
    cpp_name = "AutodiffTest"

    def test_ADFormulas(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ADFormulas")

    def test_Differentiate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Differentiate")

    def test_DifferentiateWithRequiresGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DifferentiateWithRequiresGrad")


class TestBackendTest(TestCase):
    cpp_name = "BackendTest"

    def test_ToBackend(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ToBackend")

    def test_ToBackendNotAvailable(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ToBackendNotAvailable")

    def test_TestCompiler(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCompiler")

    def test_TestComposite(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestComposite")

    def test_TestCompositeWithSetStates(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCompositeWithSetStates")

    def test_TestConsistencyOfCompositeWithSetStates(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestConsistencyOfCompositeWithSetStates"
        )

    def test_TestCompilerNotSupport(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCompilerNotSupport")


class TestBackendTestDebugInfo(TestCase):
    cpp_name = "BackendTestDebugInfo"

    def test_TestCompiler(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCompiler")

    def test_TestExceptionStackForCompilerWithModuleHierarchy(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "TestExceptionStackForCompilerWithModuleHierarchy",
        )

    def test_TestExceptionStackForCompilerWithTwoLevelModuleHierarchy(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "TestExceptionStackForCompilerWithTwoLevelModuleHierarchy",
        )

    def test_TestExceptionStackForCompilerWithLoweredSubModule(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "TestExceptionStackForCompilerWithLoweredSubModule",
        )

    def test_TestExceptionStackForCompilerWithSelectiveLoweredSubModule(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "TestExceptionStackForCompilerWithSelectiveLoweredSubModule",
        )


class TestClassImportTest(TestCase):
    cpp_name = "ClassImportTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_ScriptObject(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScriptObject")

    def test_ClassDerive(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClassDerive")

    def test_CustomClass(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CustomClass")


class TestClassParserTest(TestCase):
    cpp_name = "ClassParserTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestClassTypeTest(TestCase):
    cpp_name = "ClassTypeTest"

    def test_AddRemoveAttr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AddRemoveAttr")

    def test_AddRemoveConstant(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AddRemoveConstant")

    def test_IdenticalTypesDifferentCus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IdenticalTypesDifferentCus")


class TestTestCodeTemplate(TestCase):
    cpp_name = "TestCodeTemplate"

    def test_Copying(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Copying")

    def test_Formatting(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Formatting")


class TestConcatOptTest(TestCase):
    cpp_name = "ConcatOptTest"

    def test_SimpleCommonInputsEliminationPrefix(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleCommonInputsEliminationPrefix")

    def test_SimpleCommonInputsEliminationSuffix(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleCommonInputsEliminationSuffix")

    def test_CommonInputsEliminationWithDifferentOrderInputs(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "CommonInputsEliminationWithDifferentOrderInputs",
        )

    def test_MoreCommonInputsElimination(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoreCommonInputsElimination")

    def test_ExpandConcat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExpandConcat")

    def test_ConcatWithoutResultShape(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConcatWithoutResultShape")

    def test_ConcatWithoutInputShape(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConcatWithoutInputShape")

    def test_UseVariadicCat(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UseVariadicCat")

    def test_UseVariadicCatWithMultipleListUses(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UseVariadicCatWithMultipleListUses")

    def test_UseVariadicCatWithListMutationAfterCat(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UseVariadicCatWithListMutationAfterCat"
        )

    def test_UseVariadicCatWithListMutationBeforeCat(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UseVariadicCatWithListMutationBeforeCat"
        )

    def test_UseVariadicCatWithMultipleListMutations(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UseVariadicCatWithMultipleListMutations"
        )

    def test_RemoveListMutationUseVariadicCatAndCommonInputsElimination(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "RemoveListMutationUseVariadicCatAndCommonInputsElimination",
        )


class TestOptimizeConcatTest(TestCase):
    cpp_name = "OptimizeConcatTest"

    def test_UseVariadicCatReplaceMultiple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UseVariadicCatReplaceMultiple")


class TestConstantPoolingTest(TestCase):
    cpp_name = "ConstantPoolingTest"

    def test_Int(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Int")

    def test_PoolingAcrossBlocks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PoolingAcrossBlocks")

    def test_PoolingDifferentDevices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PoolingDifferentDevices")

    def test_DictConstantPooling(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DictConstantPooling")


class TestCleanupPassTest(TestCase):
    cpp_name = "CleanupPassTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestCreateAutodiffSubgraphsTest(TestCase):
    cpp_name = "CreateAutodiffSubgraphsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestCustomClassTest(TestCase):
    cpp_name = "CustomClassTest"

    def test_TorchbindIValueAPI(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchbindIValueAPI")

    def test_TestDocString(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDocString")

    def test_Serialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Serialization")


class TestCustomOperatorTest(TestCase):
    cpp_name = "CustomOperatorTest"

    def test_InferredSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InferredSchema")

    def test_ExplicitSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExplicitSchema")

    def test_ListParameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ListParameters")

    def test_ListParameters2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ListParameters2")

    def test_Aliasing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Aliasing")


class TestTestCustomOperator(TestCase):
    cpp_name = "TestCustomOperator"

    def test_OperatorGeneratorUndeclared(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OperatorGeneratorUndeclared")

    def test_OperatorGeneratorBasic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OperatorGeneratorBasic")


class TestEliminateDeadCodeTest(TestCase):
    cpp_name = "EliminateDeadCodeTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestFuserTest(TestCase):
    cpp_name = "FuserTest"

    def test_FusionAliasing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FusionAliasing")

    def test_KernelCaching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "KernelCaching")


class TestGraphExecutorTest(TestCase):
    cpp_name = "GraphExecutorTest"

    def test_runAsync_executor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "runAsync_executor")


class TestGraphIteratorTest(TestCase):
    cpp_name = "GraphIteratorTest"

    def test_ConstantReturnGraph(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantReturnGraph")

    def test_GraphWithParameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GraphWithParameters")

    def test_GraphWithIf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GraphWithIf")

    def test_GraphWithNestedIf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GraphWithNestedIf")

    def test_GraphWithLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GraphWithLoop")


class TestCSDebugInfoSerializaitionTest(TestCase):
    cpp_name = "CSDebugInfoSerializaitionTest"

    def test_TwoSubmodules(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TwoSubmodules")


class TestInlinerTest(TestCase):
    cpp_name = "InlinerTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestInterfaceTest(TestCase):
    cpp_name = "InterfaceTest"

    def test_ModuleInterfaceSerialization(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ModuleInterfaceSerialization")


class TestTypeCheckTest(TestCase):
    cpp_name = "TypeCheckTest"

    def test_MatchingType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MatchingType")

    def test_SizeMismatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SizeMismatch")

    def test_GradientMismatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GradientMismatch")

    def test_ScalarTypeMismatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScalarTypeMismatch")


class TestInterpreterTest(TestCase):
    cpp_name = "InterpreterTest"

    def test_IgnorableArgsInSchema(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IgnorableArgsInSchema")

    def test_IgnorableArgsInSchemaWithOut(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IgnorableArgsInSchemaWithOut")

    def test_runAsyncBasicTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "runAsyncBasicTest")


class TestEnableRethrowCaughtExceptionTest(TestCase):
    cpp_name = "EnableRethrowCaughtExceptionTest"

    def test_EnableRethrowCaughtExceptionTestRethrowsCaughtException(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "EnableRethrowCaughtExceptionTestRethrowsCaughtException",
        )


class TestIRTest(TestCase):
    cpp_name = "IRTest"

    def test_Attributes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Attributes")

    def test_Blocks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Blocks")

    def test_CommonAncestor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CommonAncestor")

    def test_OperatorMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OperatorMap")


class TestIRParserTest(TestCase):
    cpp_name = "IRParserTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_NestedBlock(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedBlock")

    def test_If(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "If")

    def test_If2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "If2")

    def test_InferredTypeIsTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InferredTypeIsTensor")

    def test_ValueReuse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ValueReuse")

    def test_Attributes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Attributes")

    def test_OptionalTypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptionalTypes")

    def test_StarTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StarTensor")

    def test_UnshapedTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnshapedTensor")

    def test_ShapedTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ShapedTensor")

    def test_NestedContrainer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedContrainer")

    def test_MalformedShapeAnnotation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MalformedShapeAnnotation")

    def test_FileCheck(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FileCheck")

    def test_Strides(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Strides")

    def test_MalformedStrides(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MalformedStrides")

    def test_TensorShapes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorShapes")

    def test_DeviceAndRequiresGradTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeviceAndRequiresGradTensors")

    def test_ListConstant(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ListConstant")

    def test_PartialStarTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PartialStarTensor")

    def test_ComplexTensorAttributes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ComplexTensorAttributes")


class TestJitTypeTest(TestCase):
    cpp_name = "JitTypeTest"

    def test_IsComplete(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsComplete")

    def test_UnifyTypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnifyTypes")


class TestLiteInterpreterTest(TestCase):
    cpp_name = "LiteInterpreterTest"

    def test_UpsampleNearest2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UpsampleNearest2d")

    def test_CheckAttrAccess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckAttrAccess")

    def test_MethodInvocation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MethodInvocation")

    def test_Conv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv")

    def test_Inline(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Inline")

    def test_Tuple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tuple")

    def test_Dict(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Dict")

    def test_PrimOverload(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrimOverload")

    def test_Prim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Prim")

    def test_PrimScalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrimScalar")

    def test_LoadOrigJit(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoadOrigJit")

    def test_WrongMethodName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WrongMethodName")

    def test_SetState(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetState")

    def test_BuiltinClass(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BuiltinClass")

    def test_BuiltinFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BuiltinFunction")

    def test_GetRuntimeByteCodeVersion(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetRuntimeByteCodeVersion")

    def test_GetByteCodeVersion(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetByteCodeVersion")

    def test_BackPortByteCodeModelAllVersions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackPortByteCodeModelAllVersions")

    def test_GetRuntimeOpsAndInfo(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetRuntimeOpsAndInfo")

    def test_isCompatibleSuccess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isCompatibleSuccess")

    def test_isCompatibleFail(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isCompatibleFail")

    def test_Eval(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Eval")

    def test_FindWrongMethodName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FindWrongMethodName")

    def test_FindAndRunMethod(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FindAndRunMethod")

    def test_RunMethodVariadic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RunMethodVariadic")

    def test_DuplicateSetState(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DuplicateSetState")

    def test_ExtraFiles(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExtraFiles")

    def test_OpNameExportFetchRootOperators(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OpNameExportFetchRootOperators")

    def test_DefaultArgsConv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DefaultArgsConv")

    def test_DefaultArgsPinv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DefaultArgsPinv")

    def test_DefaultArgsPinvSpecifyDefault(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DefaultArgsPinvSpecifyDefault")

    def test_DefaultArgsPinvWithOutArg(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DefaultArgsPinvWithOutArg")

    def test_DefaultArgsWithOutArg(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DefaultArgsWithOutArg")

    def test_TestExceptionStackWithTwoLevelModuleHierarchy(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestExceptionStackWithTwoLevelModuleHierarchy"
        )

    def test_OperatorCacheDifferentiatesDefaultArgs(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "OperatorCacheDifferentiatesDefaultArgs"
        )


class TestRunTimeTest(TestCase):
    cpp_name = "RunTimeTest"

    def test_ParseBytecode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ParseBytecode")

    def test_ParseOperator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ParseOperator")


class TestLiteTrainerTest(TestCase):
    cpp_name = "LiteTrainerTest"

    def test_Params(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Params")

    def test_SGD(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SGD")

    def test_SequentialSampler(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SequentialSampler")

    def test_RandomSamplerReturnsIndicesInCorrectRange(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RandomSamplerReturnsIndicesInCorrectRange"
        )

    def test_RandomSamplerReturnsLessValuesForLastBatch(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RandomSamplerReturnsLessValuesForLastBatch"
        )

    def test_RandomSamplerResetsWell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RandomSamplerResetsWell")

    def test_RandomSamplerResetsWithNewSizeWell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RandomSamplerResetsWithNewSizeWell")


class TestMobileTest(TestCase):
    cpp_name = "MobileTest"

    def test_SaveLoadParametersEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SaveLoadParametersEmpty")


class TestMemoryDAGTest(TestCase):
    cpp_name = "MemoryDAGTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestInternedStringsTest(TestCase):
    cpp_name = "InternedStringsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestFromQualStringTest(TestCase):
    cpp_name = "FromQualStringTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestTHNNConvTest(TestCase):
    cpp_name = "THNNConvTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestATenNativeBatchNormTest(TestCase):
    cpp_name = "ATenNativeBatchNormTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestCustomFusionTest(TestCase):
    cpp_name = "CustomFusionTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_NestedBlocks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedBlocks")


class TestControlFlowTest(TestCase):
    cpp_name = "ControlFlowTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestProtoTest(TestCase):
    cpp_name = "ProtoTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestSchemaParserTest(TestCase):
    cpp_name = "SchemaParserTest"

    def test_NestedArrays(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedArrays")

    def test_OutVariant(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OutVariant")

    def test_NamedReturns(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NamedReturns")

    def test_Futures(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Futures")

    def test_AnnotatedAliasSets(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AnnotatedAliasSets")

    def test_BeforeAfterSets(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BeforeAfterSets")

    def test_BeforeAfterSets2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BeforeAfterSets2")


class TestTopologicalIndexTest(TestCase):
    cpp_name = "TopologicalIndexTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_Reindex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reindex")


class TestRecordFunctionTest(TestCase):
    cpp_name = "RecordFunctionTest"

    def test_TracedTestInputsOutputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TracedTestInputsOutputs")

    def test_SampledCallbacks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SampledCallbacks")

    def test_RecordFunctionGuard(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RecordFunctionGuard")

    def test_Callbacks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Callbacks")

    def test_ShouldRun(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ShouldRun")

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_OperatorNameOverload(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OperatorNameOverload")


class TestThreadLocalDebugInfoTest(TestCase):
    cpp_name = "ThreadLocalDebugInfoTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestFallbackGraphsTest(TestCase):
    cpp_name = "FallbackGraphsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestNoneSchemaMatchTest(TestCase):
    cpp_name = "NoneSchemaMatchTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestPassManagementTest(TestCase):
    cpp_name = "PassManagementTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestLoopPeelerTest(TestCase):
    cpp_name = "LoopPeelerTest"

    def test_NoInductionVariableUse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoInductionVariableUse")

    def test_YesInductionVariableUse(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "YesInductionVariableUse")

    def test_LoopWithTerminationCondition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LoopWithTerminationCondition")

    def test_SimpleNestedLoops(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleNestedLoops")

    def test_SimpleNestedLoops2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleNestedLoops2")


class TestInsertAndEliminateRedundantGuardsTest(TestCase):
    cpp_name = "InsertAndEliminateRedundantGuardsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestInsertBailOutsTest(TestCase):
    cpp_name = "InsertBailOutsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestProfilerTest(TestCase):
    cpp_name = "ProfilerTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestCallStackTest(TestCase):
    cpp_name = "CallStackTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_Caching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Caching")


class TestInlinedCallStackTest(TestCase):
    cpp_name = "InlinedCallStackTest"

    def test_BlockAnnotation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BlockAnnotation")

    def test_SelfCallMethods(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SelfCallMethods")


class TestAutogradSymbolsTest(TestCase):
    cpp_name = "AutogradSymbolsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestDefaultArgTypeHintingTest(TestCase):
    cpp_name = "DefaultArgTypeHintingTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestFuturesTest(TestCase):
    cpp_name = "FuturesTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_Error(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Error")

    def test_Then(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Then")

    def test_CollectAll(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CollectAll")

    def test_CollectAny(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CollectAny")


class TestTLSFutureCallbacksTest(TestCase):
    cpp_name = "TLSFutureCallbacksTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestProfilerDisableInCallbackTest(TestCase):
    cpp_name = "ProfilerDisableInCallbackTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestRecordDebugHandles(TestCase):
    cpp_name = "RecordDebugHandles"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_ScopedCallbacks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ScopedCallbacks")


class TestIValueKWargsTest(TestCase):
    cpp_name = "IValueKWargsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestComputeFlopsTest(TestCase):
    cpp_name = "ComputeFlopsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestTestConstant(TestCase):
    cpp_name = "TestConstant"

    def test_TensorGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorGrad")


class TestTestMutation(TestCase):
    cpp_name = "TestMutation"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestTestInplaceToFunctionalActivation(TestCase):
    cpp_name = "TestInplaceToFunctionalActivation"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestTestFunctionalToInplaceActivation(TestCase):
    cpp_name = "TestFunctionalToInplaceActivation"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestMobileTypeParserTest(TestCase):
    cpp_name = "MobileTypeParserTest"

    def test_Empty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Empty")

    def test_RoundTripAnnotationStr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RoundTripAnnotationStr")

    def test_NestedContainersAnnotationStr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestedContainersAnnotationStr")

    def test_NestedContainersAnnotationStrWithSpaces(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "NestedContainersAnnotationStrWithSpaces"
        )

    def test_TypoRaises(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypoRaises")

    def test_MismatchBracketRaises(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MismatchBracketRaises")

    def test_MismatchBracketRaises2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MismatchBracketRaises2")

    def test_DictWithoutValueRaises(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DictWithoutValueRaises")

    def test_ListArgCountMismatchRaises(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ListArgCountMismatchRaises")

    def test_DictArgCountMismatchRaises(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DictArgCountMismatchRaises")

    def test_ValidTypeWithExtraStuffRaises(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ValidTypeWithExtraStuffRaises")

    def test_NonIdentifierRaises(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NonIdentifierRaises")


class TestModuleAPITest(TestCase):
    cpp_name = "ModuleAPITest"

    def test_MethodRunAsync(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MethodRunAsync")

    def test_Clone(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Clone")

    def test_CloneWithModuleInterface(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CloneWithModuleInterface")

    def test_Copy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Copy")

    def test_DeepCopy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeepCopy")

    def test_DeepCopyString(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeepCopyString")

    def test_DeepCopyPreservesAliasing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeepCopyPreservesAliasing")

    def test_Constants(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Constants")

    def test_Parameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Parameters")

    def test_Define(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Define")

    def test_Freezing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Freezing")


class TestPeepholeOptimizeTest(TestCase):
    cpp_name = "PeepholeOptimizeTest"

    def test_IsAndIsNot(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsAndIsNot")

    def test_IsAndIsNot2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsAndIsNot2")

    def test_IsAndIsNot3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsAndIsNot3")

    def test_UnwrapOptional(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnwrapOptional")

    def test_UnwrapOptional2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnwrapOptional2")

    def test_AddMMFusion(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AddMMFusion")


class TestQualifiedNameTest(TestCase):
    cpp_name = "QualifiedNameTest"

    def test_PrefixConstruction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrefixConstruction")

    def test_DottedConstruction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DottedConstruction")

    def test_BadInputRaises(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BadInputRaises")

    def test_Equality(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Equality")

    def test_IsPrefixOf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsPrefixOf")


class TestSerializationTest(TestCase):
    cpp_name = "SerializationTest"

    def test_ExtraFilesHookPreference(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExtraFilesHookPreference")

    def test_ExtraFileHooksNoSecret(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExtraFileHooksNoSecret")

    def test_ExtraFileHooksWithSecret(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExtraFileHooksWithSecret")

    def test_TypeTags(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeTags")


class TestSchemaMatchingTest(TestCase):
    cpp_name = "SchemaMatchingTest"

    def test_VarType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "VarType")

    def test_VarType2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "VarType2")


class TestStackOptTest(TestCase):
    cpp_name = "StackOptTest"

    def test_UseVariadicStack(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UseVariadicStack")

    def test_UseVariadicStackReplaceMultiple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UseVariadicStackReplaceMultiple")

    def test_UseVariadicStackWithMultipleListUses(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UseVariadicStackWithMultipleListUses")

    def test_UseVariadicStackWithListMutationAfterCat(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UseVariadicStackWithListMutationAfterCat"
        )

    def test_UseVariadicStackWithListMutationBeforeCat(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UseVariadicStackWithListMutationBeforeCat"
        )

    def test_UseVariadicStackWithMultipleListMutations(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UseVariadicStackWithMultipleListMutations"
        )


class TestSubgraphMatcherTest(TestCase):
    cpp_name = "SubgraphMatcherTest"

    def test_Trivial1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Trivial1")

    def test_Trivial2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Trivial2")

    def test_Trivial3(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Trivial3")

    def test_Trivial4(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Trivial4")

    def test_Linear1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Linear1")

    def test_Linear2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Linear2")

    def test_Diamond1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Diamond1")

    def test_Diamond2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Diamond2")

    def test_XPattern(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XPattern")

    def test_MultipleMatches(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultipleMatches")

    def test_OverlappingMatches(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OverlappingMatches")

    def test_MatchInBasicBlocks1(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MatchInBasicBlocks1")

    def test_MatchInBasicBlocks2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MatchInBasicBlocks2")

    def test_MatchesAttributes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MatchesAttributes")

    def test_BadPattern(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BadPattern")

    def test_MultiOutput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiOutput")


class TestSubgraphRewriterTest(TestCase):
    cpp_name = "SubgraphRewriterTest"

    def test_FilterMatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterMatch")

    def test_FilterNoMatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FilterNoMatch")

    def test_MultiOutput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiOutput")


class TestSubgraphUtilsTest(TestCase):
    cpp_name = "SubgraphUtilsTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_MergeSubgraphs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MergeSubgraphs")

    def test_GraphName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GraphName")


class TestUnionTypeTest(TestCase):
    cpp_name = "UnionTypeTest"

    def test_UnionOperatorEquals(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnionOperatorEquals")

    def test_UnionCreate_OptionalT1AndOptionalT2(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnionCreate_OptionalT1AndOptionalT2")

    def test_UnionCreate_OptionalTAndT(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnionCreate_OptionalTAndT")

    def test_UnionCreate_TupleWithSubtypingRelationship(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UnionCreate_TupleWithSubtypingRelationship"
        )

    def test_UnionCreate_ContainerTAndT(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnionCreate_ContainerTAndT")

    def test_UnionCreate_OptionalContainerTAndContainerTAndT(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "UnionCreate_OptionalContainerTAndContainerTAndT",
        )

    def test_Subtyping_NumberType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Subtyping_NumberType")

    def test_Subtyping_OptionalType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Subtyping_OptionalType")


class TestScriptProfileTest(TestCase):
    cpp_name = "ScriptProfileTest"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_CallingOrder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CallingOrder")


class TestJitLoggingLevelsTest(TestCase):
    cpp_name = "JitLoggingLevelsTest"

    def test_CheckSetLoggingLevel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckSetLoggingLevel")

    def test_CheckSetMultipleLogLevels(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckSetMultipleLogLevels")

    def test_CheckLoggingLevelAfterUnset(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckLoggingLevelAfterUnset")

    def test_CheckAfterChangingLevel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckAfterChangingLevel")


if __name__ == "__main__":
    run_tests()

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/test_api"


class TestAutogradAPITests(TestCase):
    cpp_name = "AutogradAPITests"

    def test_BackwardSimpleTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardSimpleTest")

    def test_BackwardTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardTest")

    def test_GradSimpleTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GradSimpleTest")

    def test_GradTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GradTest")

    def test_GradNonLeafTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GradNonLeafTest")

    def test_GradUnreachableTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GradUnreachableTest")

    def test_EmptyInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmptyInput")

    def test_RetainGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RetainGrad")

    def test_AnomalyMode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AnomalyMode")


class TestCustomAutogradTest(TestCase):
    cpp_name = "CustomAutogradTest"

    def test_GradUnreachableDiscoveryTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GradUnreachableDiscoveryTest")

    def test_CustomFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CustomFunction")

    def test_FunctionReturnsInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FunctionReturnsInput")

    def test_FunctionReturnsUndefined(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FunctionReturnsUndefined")

    def test_MaterializeGrads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaterializeGrads")

    def test_DontMaterializeGrads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DontMaterializeGrads")

    def test_NoGradCustomFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoGradCustomFunction")

    def test_MarkDirty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MarkDirty")

    def test_MarkNonDifferentiable(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MarkNonDifferentiable")

    def test_MarkNonDifferentiableMixed(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MarkNonDifferentiableMixed")

    def test_MarkNonDifferentiableNone(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MarkNonDifferentiableNone")

    def test_ReturnLeafInplace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReturnLeafInplace")

    def test_ReturnDuplicateInplace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReturnDuplicateInplace")

    def test_ReturnDuplicate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReturnDuplicate")

    def test_SaveEmptyForBackward(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SaveEmptyForBackward")

    def test_InvalidGradients(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InvalidGradients")

    def test_NoGradInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoGradInput")

    def test_TooManyGrads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TooManyGrads")

    def test_DepNoGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DepNoGrad")

    def test_Reentrant(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reentrant")

    def test_DeepReentrant(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeepReentrant")

    def test_ReentrantPriority(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReentrantPriority")

    def test_Hooks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Hooks")

    def test_HookNone(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HookNone")

    def test_BackwardWithInputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardWithInputs")

    def test_BackwardWithEmptyInputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardWithEmptyInputs")

    def test_BackwardWithNonLeafInputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardWithNonLeafInputs")

    def test_BackwardWithCreateGraphWarns(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardWithCreateGraphWarns")


class TestTestAutogradNotImplementedFallback(TestCase):
    cpp_name = "TestAutogradNotImplementedFallback"

    def test_RetSingleNonTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RetSingleNonTensor")

    def test_DoubleViewOP(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleViewOP")

    def test_InplaceOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InplaceOp")

    def test_DoubleInplaceOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DoubleInplaceOp")

    def test_OptOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptOp")

    def test_OutOfPlaceAddition(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OutOfPlaceAddition")

    def test_RetTupleNonTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RetTupleNonTensor")

    def test_ViewOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ViewOp")

    def test_RetTensorVector(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RetTensorVector")

    def test_TensorlistOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorlistOp")


class TestAnyModuleTest(TestCase):
    cpp_name = "AnyModuleTest"

    def test_SimpleReturnType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleReturnType")

    def test_SimpleReturnTypeAndSingleArgument(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleReturnTypeAndSingleArgument")

    def test_StringLiteralReturnTypeAndArgument(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StringLiteralReturnTypeAndArgument")

    def test_StringReturnTypeWithConstArgument(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StringReturnTypeWithConstArgument")

    def test_TensorReturnTypeAndStringArgumentsWithFunkyQualifications(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "TensorReturnTypeAndStringArgumentsWithFunkyQualifications",
        )

    def test_WrongArgumentType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WrongArgumentType")

    def test_WrongNumberOfArguments(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WrongNumberOfArguments")

    def test_PassingArgumentsToModuleWithDefaultArgumentsInForwardMethod(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "PassingArgumentsToModuleWithDefaultArgumentsInForwardMethod",
        )

    def test_GetWithCorrectTypeSucceeds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetWithCorrectTypeSucceeds")

    def test_GetWithIncorrectTypeThrows(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetWithIncorrectTypeThrows")

    def test_PtrWithBaseClassSucceeds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PtrWithBaseClassSucceeds")

    def test_PtrWithGoodDowncastSuccceeds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PtrWithGoodDowncastSuccceeds")

    def test_PtrWithBadDowncastThrows(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PtrWithBadDowncastThrows")

    def test_DefaultStateIsEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DefaultStateIsEmpty")

    def test_AllMethodsThrowForEmptyAnyModule(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllMethodsThrowForEmptyAnyModule")

    def test_CanMoveAssignDifferentModules(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanMoveAssignDifferentModules")

    def test_ConstructsFromModuleHolder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromModuleHolder")

    def test_ConvertsVariableToTensorCorrectly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvertsVariableToTensorCorrectly")


class TestAnyValueTest(TestCase):
    cpp_name = "AnyValueTest"

    def test_CorrectlyAccessesIntWhenCorrectType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CorrectlyAccessesIntWhenCorrectType")

    def test_CorrectlyAccessesStringLiteralWhenCorrectType(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CorrectlyAccessesStringLiteralWhenCorrectType"
        )

    def test_CorrectlyAccessesStringWhenCorrectType(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CorrectlyAccessesStringWhenCorrectType"
        )

    def test_CorrectlyAccessesPointersWhenCorrectType(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CorrectlyAccessesPointersWhenCorrectType"
        )

    def test_CorrectlyAccessesReferencesWhenCorrectType(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CorrectlyAccessesReferencesWhenCorrectType"
        )

    def test_TryGetReturnsNullptrForTheWrongType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TryGetReturnsNullptrForTheWrongType")

    def test_GetThrowsForTheWrongType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetThrowsForTheWrongType")

    def test_MoveConstructionIsAllowed(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveConstructionIsAllowed")

    def test_MoveAssignmentIsAllowed(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MoveAssignmentIsAllowed")

    def test_TypeInfoIsCorrectForInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeInfoIsCorrectForInt")

    def test_TypeInfoIsCorrectForStringLiteral(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeInfoIsCorrectForStringLiteral")

    def test_TypeInfoIsCorrectForString(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TypeInfoIsCorrectForString")


class TestDataTest(TestCase):
    cpp_name = "DataTest"

    def test_DatasetCallsGetCorrectly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DatasetCallsGetCorrectly")

    def test_TransformCallsGetApplyCorrectly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TransformCallsGetApplyCorrectly")

    def test_ChunkDataSetWithInvalidInitParameter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChunkDataSetWithInvalidInitParameter")

    def test_InfiniteStreamDataset(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InfiniteStreamDataset")

    def test_NoSequencerIsIdentity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoSequencerIsIdentity")

    def test_OrderedSequencerIsSetUpWell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OrderedSequencerIsSetUpWell")

    def test_OrderedSequencerReOrdersValues(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OrderedSequencerReOrdersValues")

    def test_BatchLambdaAppliesFunctionToBatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchLambdaAppliesFunctionToBatch")

    def test_LambdaAppliesFunctionToExample(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LambdaAppliesFunctionToExample")

    def test_CollateReducesBatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CollateReducesBatch")

    def test_CollationReducesBatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CollationReducesBatch")

    def test_SequentialSamplerReturnsIndicesInOrder(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "SequentialSamplerReturnsIndicesInOrder"
        )

    def test_SequentialSamplerReturnsLessValuesForLastBatch(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "SequentialSamplerReturnsLessValuesForLastBatch"
        )

    def test_SequentialSamplerResetsWell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SequentialSamplerResetsWell")

    def test_SequentialSamplerResetsWithNewSizeWell(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "SequentialSamplerResetsWithNewSizeWell"
        )

    def test_CanSaveAndLoadSequentialSampler(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanSaveAndLoadSequentialSampler")

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

    def test_SavingAndLoadingRandomSamplerYieldsSameSequence(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "SavingAndLoadingRandomSamplerYieldsSameSequence",
        )

    def test_StreamSamplerReturnsTheBatchSizeAndThenRemainder(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "StreamSamplerReturnsTheBatchSizeAndThenRemainder",
        )

    def test_StreamSamplerResetsWell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StreamSamplerResetsWell")

    def test_StreamSamplerResetsWithNewSizeWell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StreamSamplerResetsWithNewSizeWell")

    def test_TensorDatasetConstructsFromSingleTensor(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TensorDatasetConstructsFromSingleTensor"
        )

    def test_TensorDatasetConstructsFromInitializerListOfTensors(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "TensorDatasetConstructsFromInitializerListOfTensors",
        )

    def test_StackTransformWorksForExample(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StackTransformWorksForExample")

    def test_StackTransformWorksForTensorExample(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StackTransformWorksForTensorExample")

    def test_TensorTransformWorksForAnyTargetType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorTransformWorksForAnyTargetType")

    def test_TensorLambdaWorksforAnyTargetType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorLambdaWorksforAnyTargetType")

    def test_NormalizeTransform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NormalizeTransform")

    def test_MapDoesNotCopy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MapDoesNotCopy")

    def test_QueuePushAndPopFromSameThread(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QueuePushAndPopFromSameThread")

    def test_QueuePopWithTimeoutThrowsUponTimeout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QueuePopWithTimeoutThrowsUponTimeout")

    def test_QueuePushAndPopFromDifferentThreads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QueuePushAndPopFromDifferentThreads")

    def test_QueueClearEmptiesTheQueue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "QueueClearEmptiesTheQueue")

    def test_DataShuttleCanPushAndPopJob(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DataShuttleCanPushAndPopJob")

    def test_DataShuttleCanPushAndPopResult(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DataShuttleCanPushAndPopResult")

    def test_DataShuttlePopResultReturnsNulloptWhenNoJobsInFlight(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "DataShuttlePopResultReturnsNulloptWhenNoJobsInFlight",
        )

    def test_DataShuttleDrainMeansPopResultReturnsNullopt(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DataShuttleDrainMeansPopResultReturnsNullopt"
        )

    def test_DataShuttlePopResultTimesOut(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DataShuttlePopResultTimesOut")

    def test_SharedBatchDatasetReallyIsShared(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SharedBatchDatasetReallyIsShared")

    def test_SharedBatchDatasetDoesNotIncurCopyWhenPassedDatasetObject(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "SharedBatchDatasetDoesNotIncurCopyWhenPassedDatasetObject",
        )

    def test_CanUseCustomTypeAsIndexType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanUseCustomTypeAsIndexType")

    def test_DistributedRandomSamplerSingleReplicaProduceCorrectSamples(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "DistributedRandomSamplerSingleReplicaProduceCorrectSamples",
        )

    def test_DistributedRandomSamplerMultiReplicaProduceCorrectSamples(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "DistributedRandomSamplerMultiReplicaProduceCorrectSamples",
        )

    def test_CanSaveAndLoadDistributedRandomSampler(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CanSaveAndLoadDistributedRandomSampler"
        )

    def test_DistributedSequentialSamplerSingleReplicaProduceCorrectSamples(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "DistributedSequentialSamplerSingleReplicaProduceCorrectSamples",
        )

    def test_DistributedSequentialSamplerMultiReplicaProduceCorrectSamples(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "DistributedSequentialSamplerMultiReplicaProduceCorrectSamples",
        )

    def test_CanSaveAndLoadDistributedSequentialSampler(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CanSaveAndLoadDistributedSequentialSampler"
        )


class TestDataLoaderTest(TestCase):
    cpp_name = "DataLoaderTest"

    def test_DataLoaderOptionsDefaultAsExpected(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DataLoaderOptionsDefaultAsExpected")

    def test_DataLoaderOptionsCoalesceOptionalValues(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DataLoaderOptionsCoalesceOptionalValues"
        )

    def test_MakeDataLoaderDefaultsAsExpected(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MakeDataLoaderDefaultsAsExpected")

    def test_MakeDataLoaderThrowsWhenConstructingSamplerWithUnsizedDataset(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "MakeDataLoaderThrowsWhenConstructingSamplerWithUnsizedDataset",
        )

    def test_IteratorsCompareEqualToThemselves(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IteratorsCompareEqualToThemselves")

    def test_ValidIteratorsCompareUnequalToEachOther(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ValidIteratorsCompareUnequalToEachOther"
        )

    def test_SentinelIteratorsCompareEqualToEachOther(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "SentinelIteratorsCompareEqualToEachOther"
        )

    def test_IteratorsCompareEqualToSentinelWhenExhausted(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "IteratorsCompareEqualToSentinelWhenExhausted"
        )

    def test_IteratorsShareState(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IteratorsShareState")

    def test_CanDereferenceIteratorMultipleTimes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanDereferenceIteratorMultipleTimes")

    def test_CanUseIteratorAlgorithms(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanUseIteratorAlgorithms")

    def test_CallingBeginWhileOtherIteratorIsInFlightThrows(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CallingBeginWhileOtherIteratorIsInFlightThrows"
        )

    def test_IncrementingExhaustedValidIteratorThrows(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "IncrementingExhaustedValidIteratorThrows"
        )

    def test_DereferencingExhaustedValidIteratorThrows(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DereferencingExhaustedValidIteratorThrows"
        )

    def test_IncrementingSentinelIteratorThrows(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IncrementingSentinelIteratorThrows")

    def test_DereferencingSentinelIteratorThrows(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DereferencingSentinelIteratorThrows")

    def test_YieldsCorrectBatchSize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "YieldsCorrectBatchSize")

    def test_ReturnsLastBatchWhenSmallerThanBatchSizeWhenDropLastIsFalse(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ReturnsLastBatchWhenSmallerThanBatchSizeWhenDropLastIsFalse",
        )

    def test_DoesNotReturnLastBatchWhenSmallerThanBatchSizeWhenDropLastIsTrue(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "DoesNotReturnLastBatchWhenSmallerThanBatchSizeWhenDropLastIsTrue",
        )

    def test_RespectsTimeout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RespectsTimeout")

    def test_EnforcesOrderingAmongThreadsWhenConfigured(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "EnforcesOrderingAmongThreadsWhenConfigured"
        )

    def test_Reset(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Reset")

    def test_TestExceptionsArePropagatedFromWorkers(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestExceptionsArePropagatedFromWorkers"
        )

    def test_StatefulDatasetWithNoWorkers(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StatefulDatasetWithNoWorkers")

    def test_StatefulDatasetWithManyWorkers(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StatefulDatasetWithManyWorkers")

    def test_StatefulDatasetWithMap(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StatefulDatasetWithMap")

    def test_StatefulDatasetWithCollate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StatefulDatasetWithCollate")

    def test_ChunkDataSetGetBatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChunkDataSetGetBatch")

    def test_ChunkDataSetWithBatchSizeMismatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChunkDataSetWithBatchSizeMismatch")

    def test_ChunkDataSetWithEmptyBatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChunkDataSetWithEmptyBatch")

    def test_ChunkDataSetGetBatchWithUnevenBatchSize(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ChunkDataSetGetBatchWithUnevenBatchSize"
        )

    def test_CanAccessChunkSamplerWithChunkDataSet(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CanAccessChunkSamplerWithChunkDataSet"
        )

    def test_ChunkDatasetDoesNotHang(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChunkDatasetDoesNotHang")

    def test_ChunkDatasetSave(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChunkDatasetSave")

    def test_ChunkDatasetLoad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChunkDatasetLoad")

    def test_ChunkDatasetCrossChunkShuffle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChunkDatasetCrossChunkShuffle")

    def test_CustomPreprocessPolicy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CustomPreprocessPolicy")


class TestEnumTest(TestCase):
    cpp_name = "EnumTest"

    def test_AllEnums(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllEnums")


class TestExpandingArrayTest(TestCase):
    cpp_name = "ExpandingArrayTest"

    def test_CanConstructFromInitializerList(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanConstructFromInitializerList")

    def test_CanConstructFromVector(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanConstructFromVector")

    def test_CanConstructFromArray(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanConstructFromArray")

    def test_CanConstructFromSingleValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanConstructFromSingleValue")

    def test_ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInInitializerList(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInInitializerList",
        )

    def test_ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInVector(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInVector",
        )


class TestFFTTest(TestCase):
    cpp_name = "FFTTest"

    def test_fft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fft")

    def test_fft_real(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fft_real")

    def test_fft_pad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fft_pad")

    def test_fft_norm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fft_norm")

    def test_ifft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ifft")

    def test_fft_ifft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "fft_ifft")

    def test_rfft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "rfft")

    def test_rfft_irfft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "rfft_irfft")

    def test_ihfft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ihfft")

    def test_hfft_ihfft(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "hfft_ihfft")


class TestFunctionalTest(TestCase):
    cpp_name = "FunctionalTest"

    def test_Conv1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv1d")

    def test_Conv2dEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2dEven")

    def test_Conv2dUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2dUneven")

    def test_Conv3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv3d")

    def test_MaxPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool1d")

    def test_MaxPool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool2d")

    def test_MaxPool2dBackward(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool2dBackward")

    def test_MaxPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool3d")

    def test_AvgPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AvgPool1d")

    def test_AvgPool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AvgPool2d")

    def test_AvgPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AvgPool3d")

    def test_FractionalMaxPool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FractionalMaxPool2d")

    def test_FractionalMaxPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FractionalMaxPool3d")

    def test_LPPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LPPool1d")

    def test_LPPool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LPPool2d")

    def test_CosineSimilarity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CosineSimilarity")

    def test_SmoothL1LossDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SmoothL1LossDefaultOptions")

    def test_SmoothL1LossBeta(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SmoothL1LossBeta")

    def test_SmoothL1LossNoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SmoothL1LossNoReduction")

    def test_HuberLossDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HuberLossDefaultOptions")

    def test_HuberLossDelta(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HuberLossDelta")

    def test_HuberLossNoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HuberLossNoReduction")

    def test_SoftMarginLossDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SoftMarginLossDefaultOptions")

    def test_MultiLabelSoftMarginLossDefaultOptions(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MultiLabelSoftMarginLossDefaultOptions"
        )

    def test_SoftMarginLossNoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SoftMarginLossNoReduction")

    def test_MultiLabelSoftMarginLossWeightedNoReduction(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MultiLabelSoftMarginLossWeightedNoReduction"
        )

    def test_PairwiseDistance(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PairwiseDistance")

    def test_PDist(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PDist")

    def test_AdaptiveMaxPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool1d")

    def test_AdaptiveMaxPool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool2d")

    def test_AdaptiveMaxPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool3d")

    def test_AdaptiveAvgPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveAvgPool1d")

    def test_AdaptiveAvgPool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveAvgPool2d")

    def test_AdaptiveAvgPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveAvgPool3d")

    def test_L1Loss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "L1Loss")

    def test_MSELoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MSELoss")

    def test_BCELoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BCELoss")

    def test_KLDivLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "KLDivLoss")

    def test_HingeEmbeddingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HingeEmbeddingLoss")

    def test_GridSample(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GridSample")

    def test_AffineGrid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AffineGrid")

    def test_MultiMarginLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiMarginLoss")

    def test_CosineEmbeddingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CosineEmbeddingLoss")

    def test_MultiLabelMarginLossDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiLabelMarginLossDefaultOptions")

    def test_MultiLabelMarginLossNoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiLabelMarginLossNoReduction")

    def test_TripletMarginLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TripletMarginLoss")

    def test_TripletMarginWithDistanceLossDefaultParity(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TripletMarginWithDistanceLossDefaultParity"
        )

    def test_NLLLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NLLLoss")

    def test_CrossEntropy(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CrossEntropy")

    def test_MaxUnpool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxUnpool1d")

    def test_MaxUnpool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxUnpool2d")

    def test_MaxUnpool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxUnpool3d")

    def test_ELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ELU")

    def test_SELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SELU")

    def test_GLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GLU")

    def test_GELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GELU")

    def test_Hardshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Hardshrink")

    def test_OneHot(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OneHot")

    def test_Hardtanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Hardtanh")

    def test_LeakyReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LeakyReLU")

    def test_LogSigmoid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LogSigmoid")

    def test_GumbelSoftmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GumbelSoftmax")

    def test_Softmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmax")

    def test_Softmin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmin")

    def test_LogSoftmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LogSoftmax")

    def test_PReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PReLU")

    def test_LayerNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LayerNorm")

    def test_GroupNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GroupNorm")

    def test_LocalResponseNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LocalResponseNorm")

    def test_Linear(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Linear")

    def test_Embedding(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Embedding")

    def test_EmbeddingBag(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmbeddingBag")

    def test_Bilinear(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bilinear")

    def test_Normalize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Normalize")

    def test_ReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReLU")

    def test_ReLUDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReLUDefaultOptions")

    def test_ReLU6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReLU6")

    def test_ReLU6DefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReLU6DefaultOptions")

    def test_RReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RReLU")

    def test_RReLUDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RReLUDefaultOptions")

    def test_CELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CELU")

    def test_CELUDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CELUDefaultOptions")

    def test_PixelShuffle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PixelShuffle")

    def test_PixelUnshuffle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PixelUnshuffle")

    def test_Softplus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softplus")

    def test_SoftplusDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SoftplusDefaultOptions")

    def test_Fold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Fold")

    def test_Unfold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Unfold")

    def test_Softshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softshrink")

    def test_SoftshrinkDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SoftshrinkDefaultOptions")

    def test_Softsign(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softsign")

    def test_Mish(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Mish")

    def test_Tanhshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tanhshrink")

    def test_Threshold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Threshold")

    def test_BatchNorm1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm1d")

    def test_BatchNorm1dDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm1dDefaultOptions")

    def test_BatchNorm2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm2d")

    def test_BatchNorm2dDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm2dDefaultOptions")

    def test_BatchNorm3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm3d")

    def test_BatchNorm3dDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm3dDefaultOptions")

    def test_InstanceNorm1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm1d")

    def test_InstanceNorm1dDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm1dDefaultOptions")

    def test_InstanceNorm2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm2d")

    def test_InstanceNorm2dDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm2dDefaultOptions")

    def test_InstanceNorm3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm3d")

    def test_InstanceNorm3dDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm3dDefaultOptions")

    def test_Interpolate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Interpolate")

    def test_Pad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Pad")

    def test_CTCLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CTCLoss")

    def test_PoissonNLLLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PoissonNLLLoss")

    def test_MarginRankingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MarginRankingLoss")

    def test_ConvTranspose1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose1d")

    def test_ConvTranspose2dEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose2dEven")

    def test_ConvTranspose2dUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose2dUneven")

    def test_ConvTranspose3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose3d")

    def test_AlphaDropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AlphaDropout")

    def test_FeatureAlphaDropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FeatureAlphaDropout")

    def test_Dropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Dropout")

    def test_Dropout2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Dropout2d")

    def test_Dropout3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Dropout3d")

    def test_isfinite(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isfinite")

    def test_isinf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isinf")

    def test_AllClose(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllClose")

    def test_BCEWithLogitsLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BCEWithLogitsLoss")


class TestIntegrationTest(TestCase):
    cpp_name = "IntegrationTest"

    def test_CartPole(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CartPole")


class TestInitTest(TestCase):
    cpp_name = "InitTest"

    def test_ProducesPyTorchValues_XavierUniform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_XavierUniform")

    def test_ProducesPyTorchValues_XavierNormal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_XavierNormal")

    def test_ProducesPyTorchValues_KaimingNormal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_KaimingNormal")

    def test_ProducesPyTorchValues_KaimingUniform(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_KaimingUniform")

    def test_CanInitializeTensorThatRequiresGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanInitializeTensorThatRequiresGrad")

    def test_CalculateGainWithTanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CalculateGainWithTanh")

    def test_CalculateGainWithRelu(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CalculateGainWithRelu")

    def test_CalculateGainWithLeakyRelu(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CalculateGainWithLeakyRelu")

    def test_CanInitializeCnnWithOrthogonal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanInitializeCnnWithOrthogonal")


class TestTorchScriptTest(TestCase):
    cpp_name = "TorchScriptTest"

    def test_CanCompileMultipleFunctions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanCompileMultipleFunctions")

    def test_TestNestedIValueModuleArgMatching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNestedIValueModuleArgMatching")

    def test_TestDictArgMatching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDictArgMatching")

    def test_TestTupleArgMatching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestTupleArgMatching")

    def test_TestOptionalArgMatching(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestOptionalArgMatching")

    def test_TestPickle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestPickle")


class TestMakeUniqueTest(TestCase):
    cpp_name = "MakeUniqueTest"

    def test_ForwardRvaluesCorrectly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ForwardRvaluesCorrectly")

    def test_ForwardLvaluesCorrectly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ForwardLvaluesCorrectly")

    def test_CanConstructUniquePtrOfArray(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanConstructUniquePtrOfArray")


class TestMetaTensorTest(TestCase):
    cpp_name = "MetaTensorTest"

    def test_MetaDeviceApi(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MetaDeviceApi")

    def test_MetaNamespaceApi(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MetaNamespaceApi")


class TestUtilsTest(TestCase):
    cpp_name = "UtilsTest"

    def test_WarnOnce(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WarnOnce")

    def test_AmbiguousOperatorDefaults(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AmbiguousOperatorDefaults")


class TestNoGradTest(TestCase):
    cpp_name = "NoGradTest"

    def test_SetsGradModeCorrectly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetsGradModeCorrectly")


class TestAutogradTest(TestCase):
    cpp_name = "AutogradTest"

    def test_CanTakeDerivatives(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanTakeDerivatives")

    def test_CanTakeDerivativesOfZeroDimTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanTakeDerivativesOfZeroDimTensors")

    def test_CanPassCustomGradientInputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanPassCustomGradientInputs")


class TestModuleTest(TestCase):
    cpp_name = "ModuleTest"

    def test_CanEnableAndDisableTrainingMode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanEnableAndDisableTrainingMode")

    def test_ZeroGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ZeroGrad")

    def test_ZeroGradWithUndefined(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ZeroGradWithUndefined")

    def test_RegisterModuleThrowsForEmptyOrDottedName(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterModuleThrowsForEmptyOrDottedName"
        )

    def test_RegisterModuleThrowsForDuplicateModuleName(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterModuleThrowsForDuplicateModuleName"
        )

    def test_ReplaceModuleThrowsForUnknownModuleName(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ReplaceModuleThrowsForUnknownModuleName"
        )

    def test_ReplaceModule(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReplaceModule")

    def test_UnregisterModule(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UnregisterModule")

    def test_RegisterParameterThrowsForEmptyOrDottedName(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterParameterThrowsForEmptyOrDottedName"
        )

    def test_RegisterParameterThrowsForDuplicateModuleName(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterParameterThrowsForDuplicateModuleName"
        )

    def test_RegisterParameterUndefinedTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegisterParameterUndefinedTensor")

    def test_RegisterBufferThrowsForEmptyOrDottedName(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterBufferThrowsForEmptyOrDottedName"
        )

    def test_RegisterBufferThrowsForDuplicateModuleName(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "RegisterBufferThrowsForDuplicateModuleName"
        )

    def test_CanGetName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanGetName")

    def test_AsCastsModulesCorrectly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AsCastsModulesCorrectly")

    def test_DeviceOrDtypeConversionSkipsUndefinedTensor(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "DeviceOrDtypeConversionSkipsUndefinedTensor"
        )

    def test_ParametersAndBuffersAccessorSkipsUndefinedTensor(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ParametersAndBuffersAccessorSkipsUndefinedTensor",
        )

    def test_CallingCloneOnModuleThatDoesNotOverrideCloneThrows(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "CallingCloneOnModuleThatDoesNotOverrideCloneThrows",
        )

    def test_CallingCloneOnModuleThatDoesOverrideCloneDoesNotThrow(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "CallingCloneOnModuleThatDoesOverrideCloneDoesNotThrow",
        )

    def test_CloneCreatesDistinctParameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CloneCreatesDistinctParameters")

    def test_ClonePreservesExternalReferences(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClonePreservesExternalReferences")

    def test_CloneCopiesTheValuesOfVariablesOfSubmodules(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CloneCopiesTheValuesOfVariablesOfSubmodules"
        )

    def test_HasCorrectNumberOfParameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HasCorrectNumberOfParameters")

    def test_ContainsParametersWithTheCorrectName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ContainsParametersWithTheCorrectName")

    def test_HasCorrectNumberOfBuffers(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HasCorrectNumberOfBuffers")

    def test_ContainsBuffersWithTheCorrectName(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ContainsBuffersWithTheCorrectName")

    def test_DefaultConstructorOfModuleHolderCallsDefaultConstructorOfImpl(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "DefaultConstructorOfModuleHolderCallsDefaultConstructorOfImpl",
        )

    def test_ValueConstructorOfModuleHolderCallsCorrectConstructorInImpl(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ValueConstructorOfModuleHolderCallsCorrectConstructorInImpl",
        )

    def test_NullptrConstructorLeavesTheModuleHolderInEmptyState(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "NullptrConstructorLeavesTheModuleHolderInEmptyState",
        )

    def test_ModulesReturnsExpectedSubmodulesForFlatModel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ModulesReturnsExpectedSubmodulesForFlatModel"
        )

    def test_ModulesExcludesSelfWhenIncludeSelfSetToFalse(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ModulesExcludesSelfWhenIncludeSelfSetToFalse"
        )

    def test_NamedModulesReturnsExpectedNamedSubmodulesForFlatModel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "NamedModulesReturnsExpectedNamedSubmodulesForFlatModel",
        )

    def test_NamedModulesExcludesSelfWhenIncludeSelfSetToFalse(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "NamedModulesExcludesSelfWhenIncludeSelfSetToFalse",
        )

    def test_ChildrenReturnsExpectedSubmodulesForFlatModel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ChildrenReturnsExpectedSubmodulesForFlatModel"
        )

    def test_NamedChildrenReturnsExpectedNamedSubmodulesForFlatModel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "NamedChildrenReturnsExpectedNamedSubmodulesForFlatModel",
        )

    def test_ParametersReturnsExpectedTensorsForFlatModel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ParametersReturnsExpectedTensorsForFlatModel"
        )

    def test_NamedParametersReturnsExpectedTensorsForFlatModel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "NamedParametersReturnsExpectedTensorsForFlatModel",
        )

    def test_BuffersReturnsExpectedTensorsForFlatModel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "BuffersReturnsExpectedTensorsForFlatModel"
        )

    def test_NamedBuffersReturnsExpectedTensorsForFlatModel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "NamedBuffersReturnsExpectedTensorsForFlatModel"
        )

    def test_ModulesReturnsExpectedSubmodulesForDeepModel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ModulesReturnsExpectedSubmodulesForDeepModel"
        )

    def test_NamedModulesReturnsExpectedNamedSubmodulesForDeepModel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "NamedModulesReturnsExpectedNamedSubmodulesForDeepModel",
        )

    def test_ChildrensReturnsExpectedSubmodulesForDeepModel(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ChildrensReturnsExpectedSubmodulesForDeepModel"
        )

    def test_NamedChildrensReturnsExpectedNamedSubmodulesForDeepModel(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "NamedChildrensReturnsExpectedNamedSubmodulesForDeepModel",
        )

    def test_ModuleApplyIteratesCorreclty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ModuleApplyIteratesCorreclty")

    def test_ConstModuleApplyIteratesCorreclty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstModuleApplyIteratesCorreclty")

    def test_NamedModuleApplyIteratesCorreclty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NamedModuleApplyIteratesCorreclty")

    def test_ConstNamedModuleApplyIteratesCorreclty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ConstNamedModuleApplyIteratesCorreclty"
        )

    def test_ModulePointerApplyIteratesCorreclty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ModulePointerApplyIteratesCorreclty")

    def test_NamedModulePointerApplyIteratesCorreclty(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "NamedModulePointerApplyIteratesCorreclty"
        )

    def test_ThrowsWhenAttemptingtoGetTopLevelModuleAsSharedPtr(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ThrowsWhenAttemptingtoGetTopLevelModuleAsSharedPtr",
        )

    def test_PrettyPrint(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrint")

    def test_CanCallForwardOnNonTensorForwardThroughPimpl(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CanCallForwardOnNonTensorForwardThroughPimpl"
        )


class TestModuleDictTest(TestCase):
    cpp_name = "ModuleDictTest"

    def test_ConstructsFromList(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromList")

    def test_ConstructsFromordereddict(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromordereddict")

    def test_UpdatePopClearContains(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UpdatePopClearContains")

    def test_UpdateExist(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UpdateExist")

    def test_Keys(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Keys")

    def test_Values(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Values")

    def test_SanityCheckForHoldingStandardModules(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SanityCheckForHoldingStandardModules")

    def test_HasReferenceSemantics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HasReferenceSemantics")

    def test_IsCloneable(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsCloneable")

    def test_RegistersElementsAsSubmodules(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegistersElementsAsSubmodules")

    def test_PrettyPrintModuleDict(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintModuleDict")


class TestModuleListTest(TestCase):
    cpp_name = "ModuleListTest"

    def test_ConstructsFromSharedPointer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromSharedPointer")

    def test_ConstructsFromConcreteType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromConcreteType")

    def test_ConstructsFromModuleHolder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromModuleHolder")

    def test_PushBackAddsAnElement(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PushBackAddsAnElement")

    def test_Insertion(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Insertion")

    def test_AccessWithAt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AccessWithAt")

    def test_AccessWithPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AccessWithPtr")

    def test_SanityCheckForHoldingStandardModules(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SanityCheckForHoldingStandardModules")

    def test_ExtendPushesModulesFromOtherModuleList(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ExtendPushesModulesFromOtherModuleList"
        )

    def test_HasReferenceSemantics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HasReferenceSemantics")

    def test_IsCloneable(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsCloneable")

    def test_RegistersElementsAsSubmodules(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegistersElementsAsSubmodules")

    def test_NestingIsPossible(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NestingIsPossible")

    def test_PrettyPrintModuleList(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintModuleList")

    def test_RangeBasedForLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RangeBasedForLoop")


class TestModulesTest(TestCase):
    cpp_name = "ModulesTest"

    def test_Conv1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv1d")

    def test_Conv1dSameStrided(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv1dSameStrided")

    def test_Conv2dEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2dEven")

    def test_Conv2dUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2dUneven")

    def test_Conv2dSameStrided(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv2dSameStrided")

    def test_Conv3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv3d")

    def test_Conv3dSameStrided(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Conv3dSameStrided")

    def test_ConvTranspose1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose1d")

    def test_ConvTranspose2dEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose2dEven")

    def test_ConvTranspose2dUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose2dUneven")

    def test_ConvTranspose3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvTranspose3d")

    def test_MaxPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool1d")

    def test_MaxPool1dReturnIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool1dReturnIndices")

    def test_MaxPool2dEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool2dEven")

    def test_MaxPool2dUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool2dUneven")

    def test_MaxPool2dReturnIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool2dReturnIndices")

    def test_MaxPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool3d")

    def test_MaxPool3dReturnIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool3dReturnIndices")

    def test_AvgPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AvgPool1d")

    def test_AvgPool2dEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AvgPool2dEven")

    def test_AvgPool2dUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AvgPool2dUneven")

    def test_AvgPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AvgPool3d")

    def test_FractionalMaxPool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FractionalMaxPool2d")

    def test_FractionalMaxPool2dReturnIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FractionalMaxPool2dReturnIndices")

    def test_FractionalMaxPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FractionalMaxPool3d")

    def test_FractionalMaxPool3dReturnIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FractionalMaxPool3dReturnIndices")

    def test_LPPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LPPool1d")

    def test_LPPool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LPPool2d")

    def test_Identity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Identity")

    def test_Flatten(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Flatten")

    def test_Unflatten(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Unflatten")

    def test_AdaptiveMaxPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool1d")

    def test_AdaptiveMaxPool1dReturnIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool1dReturnIndices")

    def test_AdaptiveMaxPool2dEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool2dEven")

    def test_AdaptiveMaxPool2dUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool2dUneven")

    def test_AdaptiveMaxPool2dReturnIndicesEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool2dReturnIndicesEven")

    def test_AdaptiveMaxPool2dReturnIndicesUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool2dReturnIndicesUneven")

    def test_AdaptiveMaxPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool3d")

    def test_AdaptiveMaxPool3dReturnIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveMaxPool3dReturnIndices")

    def test_AdaptiveAvgPool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveAvgPool1d")

    def test_AdaptiveAvgPool2dEven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveAvgPool2dEven")

    def test_AdaptiveAvgPool2dUneven(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveAvgPool2dUneven")

    def test_AdaptiveAvgPool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveAvgPool3d")

    def test_MaxUnpool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxUnpool1d")

    def test_MaxPool1d_MaxUnpool1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool1d_MaxUnpool1d")

    def test_MaxUnpool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxUnpool2d")

    def test_MaxPool2d_MaxUnpool2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool2d_MaxUnpool2d")

    def test_MaxUnpool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxUnpool3d")

    def test_MaxUnpool3dOutputSize(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxUnpool3dOutputSize")

    def test_MaxPool3d_MaxUnpool3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MaxPool3d_MaxUnpool3d")

    def test_Linear(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Linear")

    def test_LocalResponseNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LocalResponseNorm")

    def test_LayerNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LayerNorm")

    def test_GroupNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GroupNorm")

    def test_Bilinear(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Bilinear")

    def test_Fold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Fold")

    def test_Unfold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Unfold")

    def test_SimpleContainer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleContainer")

    def test_EmbeddingBasic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmbeddingBasic")

    def test_EmbeddingList(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmbeddingList")

    def test_EmbeddingFromPretrained(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmbeddingFromPretrained")

    def test_EmbeddingBagFromPretrained(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EmbeddingBagFromPretrained")

    def test_AlphaDropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AlphaDropout")

    def test_FeatureAlphaDropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FeatureAlphaDropout")

    def test_Dropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Dropout")

    def test_Dropout2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Dropout2d")

    def test_Dropout3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Dropout3d")

    def test_Parameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Parameters")

    def test_FunctionalCallsSuppliedFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FunctionalCallsSuppliedFunction")

    def test_FunctionalWithTorchFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FunctionalWithTorchFunction")

    def test_FunctionalArgumentBinding(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FunctionalArgumentBinding")

    def test_BatchNorm1dStateful(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm1dStateful")

    def test_BatchNorm1dStateless(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm1dStateless")

    def test_BatchNorm1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm1d")

    def test_BatchNorm2dStateful(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm2dStateful")

    def test_BatchNorm2dStateless(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm2dStateless")

    def test_BatchNorm2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm2d")

    def test_BatchNorm3dStateful(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm3dStateful")

    def test_BatchNorm3dStateless(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm3dStateless")

    def test_BatchNorm3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BatchNorm3d")

    def test_InstanceNorm1dStateful(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm1dStateful")

    def test_InstanceNorm1dStateless(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm1dStateless")

    def test_InstanceNorm1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm1d")

    def test_InstanceNorm2dStateful(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm2dStateful")

    def test_InstanceNorm2dStateless(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm2dStateless")

    def test_InstanceNorm2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm2d")

    def test_InstanceNorm3dStateful(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm3dStateful")

    def test_InstanceNorm3dStateless(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm3dStateless")

    def test_InstanceNorm3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InstanceNorm3d")

    def test_L1Loss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "L1Loss")

    def test_MSELoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MSELoss")

    def test_BCELoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BCELoss")

    def test_KLDivLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "KLDivLoss")

    def test_HingeEmbeddingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HingeEmbeddingLoss")

    def test_MultiMarginLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiMarginLoss")

    def test_CosineEmbeddingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CosineEmbeddingLoss")

    def test_SmoothL1LossDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SmoothL1LossDefaultOptions")

    def test_HuberLossDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HuberLossDefaultOptions")

    def test_MultiLabelMarginLossDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiLabelMarginLossDefaultOptions")

    def test_SmoothL1LossNoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SmoothL1LossNoReduction")

    def test_HuberLossNoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HuberLossNoReduction")

    def test_MultiLabelMarginLossNoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiLabelMarginLossNoReduction")

    def test_SmoothL1LossBeta(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SmoothL1LossBeta")

    def test_HuberLossDelta(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HuberLossDelta")

    def test_TripletMarginLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TripletMarginLoss")

    def test_TripletMarginWithDistanceLossDefaultParity(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TripletMarginWithDistanceLossDefaultParity"
        )

    def test_TripletMarginWithDistanceLossFunctionalParity(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TripletMarginWithDistanceLossFunctionalParity"
        )

    def test_NLLLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NLLLoss")

    def test_CrossEntropyLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CrossEntropyLoss")

    def test_CosineSimilarity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CosineSimilarity")

    def test_SoftMarginLossDefaultOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SoftMarginLossDefaultOptions")

    def test_MultiLabelSoftMarginLossDefaultOptions(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MultiLabelSoftMarginLossDefaultOptions"
        )

    def test_SoftMarginLossNoReduction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SoftMarginLossNoReduction")

    def test_MultiLabelSoftMarginLossWeightedNoReduction(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "MultiLabelSoftMarginLossWeightedNoReduction"
        )

    def test_PairwiseDistance(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PairwiseDistance")

    def test_ELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ELU")

    def test_SELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SELU")

    def test_Hardshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Hardshrink")

    def test_Hardtanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Hardtanh")

    def test_HardtanhMinValGEMaxVal(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HardtanhMinValGEMaxVal")

    def test_LeakyReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LeakyReLU")

    def test_LogSigmoid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LogSigmoid")

    def test_Softmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmax")

    def test_Softmin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmin")

    def test_LogSoftmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LogSoftmax")

    def test_AdaptiveLogSoftmaxWithLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AdaptiveLogSoftmaxWithLoss")

    def test_Softmax2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softmax2d")

    def test_PReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PReLU")

    def test_ReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReLU")

    def test_ReLU6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReLU6")

    def test_RReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RReLU")

    def test_CELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CELU")

    def test_GLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GLU")

    def test_GELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GELU")

    def test_Mish(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Mish")

    def test_Sigmoid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sigmoid")

    def test_PixelShuffle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PixelShuffle")

    def test_PixelUnshuffle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PixelUnshuffle")

    def test_Softplus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softplus")

    def test_Softshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softshrink")

    def test_Softsign(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Softsign")

    def test_Tanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tanh")

    def test_Tanhshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Tanhshrink")

    def test_Threshold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Threshold")

    def test_Upsampling1D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Upsampling1D")

    def test_Upsampling2D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Upsampling2D")

    def test_Upsampling3D(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Upsampling3D")

    def test_CTCLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CTCLoss")

    def test_PoissonNLLLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PoissonNLLLoss")

    def test_MarginRankingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MarginRankingLoss")

    def test_BCEWithLogitsLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BCEWithLogitsLoss")

    def test_MultiheadAttention(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultiheadAttention")

    def test_PrettyPrintIdentity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintIdentity")

    def test_PrettyPrintFlatten(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintFlatten")

    def test_PrettyPrintUnflatten(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintUnflatten")

    def test_ReflectionPad1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReflectionPad1d")

    def test_ReflectionPad2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReflectionPad2d")

    def test_ReflectionPad3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReflectionPad3d")

    def test_ReplicationPad1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReplicationPad1d")

    def test_ReplicationPad2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReplicationPad2d")

    def test_ReplicationPad3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReplicationPad3d")

    def test_ZeroPad2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ZeroPad2d")

    def test_ConstantPad1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantPad1d")

    def test_ConstantPad2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantPad2d")

    def test_ConstantPad3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstantPad3d")

    def test_CrossMapLRN2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CrossMapLRN2d")

    def test_RNNCell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RNNCell")

    def test_LSTMCell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "LSTMCell")

    def test_GRUCell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GRUCell")

    def test_PrettyPrintLinear(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintLinear")

    def test_PrettyPrintBilinear(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintBilinear")

    def test_PrettyPrintConv(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintConv")

    def test_PrettyPrintConvTranspose(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintConvTranspose")

    def test_PrettyPrintUpsample(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintUpsample")

    def test_PrettyPrintFold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintFold")

    def test_PrettyPrintUnfold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintUnfold")

    def test_PrettyPrintMaxPool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintMaxPool")

    def test_PrettyPrintAvgPool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintAvgPool")

    def test_PrettyPrinFractionalMaxPool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrinFractionalMaxPool")

    def test_PrettyPrintLPPool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintLPPool")

    def test_PrettyPrintAdaptiveMaxPool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintAdaptiveMaxPool")

    def test_PrettyPrintAdaptiveAvgPool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintAdaptiveAvgPool")

    def test_PrettyPrintMaxUnpool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintMaxUnpool")

    def test_PrettyPrintDropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintDropout")

    def test_PrettyPrintDropout2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintDropout2d")

    def test_PrettyPrintDropout3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintDropout3d")

    def test_PrettyPrintFunctional(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintFunctional")

    def test_PrettyPrintBatchNorm1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintBatchNorm1d")

    def test_PrettyPrintBatchNorm2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintBatchNorm2d")

    def test_PrettyPrintBatchNorm3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintBatchNorm3d")

    def test_PrettyPrintInstanceNorm1d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintInstanceNorm1d")

    def test_PrettyPrintInstanceNorm2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintInstanceNorm2d")

    def test_PrettyPrintInstanceNorm3d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintInstanceNorm3d")

    def test_PrettyPrintLayerNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintLayerNorm")

    def test_PrettyPrintGroupNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintGroupNorm")

    def test_PrettyPrintLocalResponseNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintLocalResponseNorm")

    def test_PrettyPrintEmbedding(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintEmbedding")

    def test_PrettyPrintEmbeddingBag(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintEmbeddingBag")

    def test_PrettyPrintL1Loss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintL1Loss")

    def test_PrettyPrintKLDivLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintKLDivLoss")

    def test_PrettyPrintMSELoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintMSELoss")

    def test_PrettyPrintBCELoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintBCELoss")

    def test_PrettyPrintHingeEmbeddingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintHingeEmbeddingLoss")

    def test_PrettyPrintCosineEmbeddingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintCosineEmbeddingLoss")

    def test_PrettyPrintTripletMarginLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintTripletMarginLoss")

    def test_PrettyPrintTripletMarginWithDistanceLoss(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "PrettyPrintTripletMarginWithDistanceLoss"
        )

    def test_PrettyPrintNLLLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintNLLLoss")

    def test_PrettyPrinCrossEntropyLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrinCrossEntropyLoss")

    def test_PrettyPrintMultiLabelMarginLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintMultiLabelMarginLoss")

    def test_PrettyPrintMultiLabelSoftMarginLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintMultiLabelSoftMarginLoss")

    def test_PrettyPrintSoftMarginLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSoftMarginLoss")

    def test_PrettyPrintCosineSimilarity(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintCosineSimilarity")

    def test_PrettyPrintPairwiseDistance(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintPairwiseDistance")

    def test_PrettyPrintReflectionPad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintReflectionPad")

    def test_PrettyPrintReplicationPad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintReplicationPad")

    def test_PrettyPrintZeroPad2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintZeroPad2d")

    def test_PrettyPrintConstantPad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintConstantPad")

    def test_PrettyPrintNestedModel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintNestedModel")

    def test_PrettyPrintELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintELU")

    def test_PrettyPrintSELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSELU")

    def test_PrettyPrintGLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintGLU")

    def test_PrettyPrintHardshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintHardshrink")

    def test_PrettyPrintHardtanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintHardtanh")

    def test_PrettyPrintLeakyReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintLeakyReLU")

    def test_PrettyPrintLogSigmoid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintLogSigmoid")

    def test_PrettyPrintSoftmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSoftmax")

    def test_PrettyPrintSoftmin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSoftmin")

    def test_PrettyPrintLogSoftmax(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintLogSoftmax")

    def test_PrettyPrintSoftmax2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSoftmax2d")

    def test_PrettyPrintPReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintPReLU")

    def test_PrettyPrintReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintReLU")

    def test_PrettyPrintReLU6(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintReLU6")

    def test_PrettyPrintRReLU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintRReLU")

    def test_PrettyPrintCELU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintCELU")

    def test_PrettyPrintSigmoid(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSigmoid")

    def test_PrettyPrintPixelShuffle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintPixelShuffle")

    def test_PrettyPrintPixelUnshuffle(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintPixelUnshuffle")

    def test_PrettyPrintSoftplus(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSoftplus")

    def test_PrettyPrintSoftshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSoftshrink")

    def test_PrettyPrintSoftsign(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSoftsign")

    def test_PrettyPrintTanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintTanh")

    def test_PrettyPrintTanhshrink(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintTanhshrink")

    def test_PrettyPrintThreshold(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintThreshold")

    def test_PrettyPrintCTCLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintCTCLoss")

    def test_PrettyPrintPoissonNLLLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintPoissonNLLLoss")

    def test_PrettyPrintMarginRankingLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintMarginRankingLoss")

    def test_PrettyPrintCrossMapLRN2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintCrossMapLRN2d")

    def test_PrettyPrintAlphaDropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintAlphaDropout")

    def test_PrettyPrintFeatureAlphaDropout(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintFeatureAlphaDropout")

    def test_PrettyPrintBCEWithLogitsLoss(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintBCEWithLogitsLoss")

    def test_PrettyPrintMultiheadAttention(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintMultiheadAttention")

    def test_PrettyPrintRNNCell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintRNNCell")

    def test_PrettyPrintLSTMCell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintLSTMCell")

    def test_PrettyPrintGRUCell(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintGRUCell")

    def test_PrettyPrintAdaptiveLogSoftmaxWithLoss(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "PrettyPrintAdaptiveLogSoftmaxWithLoss"
        )


class TestParameterDictTest(TestCase):
    cpp_name = "ParameterDictTest"

    def test_ConstructFromTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructFromTensor")

    def test_ConstructFromOrderedDict(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructFromOrderedDict")

    def test_InsertAndContains(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InsertAndContains")

    def test_InsertAndClear(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InsertAndClear")

    def test_InsertAndPop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "InsertAndPop")

    def test_SimpleUpdate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SimpleUpdate")

    def test_Keys(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Keys")

    def test_Values(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Values")

    def test_Get(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Get")

    def test_PrettyPrintParameterDict(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintParameterDict")


class TestParameterListTest(TestCase):
    cpp_name = "ParameterListTest"

    def test_ConstructsFromSharedPointer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromSharedPointer")

    def test_isEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "isEmpty")

    def test_PushBackAddsAnElement(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PushBackAddsAnElement")

    def test_ForEachLoop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ForEachLoop")

    def test_AccessWithAt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AccessWithAt")

    def test_ExtendPushesParametersFromOtherParameterList(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ExtendPushesParametersFromOtherParameterList"
        )

    def test_PrettyPrintParameterList(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintParameterList")

    def test_IncrementAdd(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IncrementAdd")


class TestNamespaceTests(TestCase):
    cpp_name = "NamespaceTests"

    def test_NotLeakingSymbolsFromTorchAutogradNamespace(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "NotLeakingSymbolsFromTorchAutogradNamespace"
        )


class TestNNUtilsTest(TestCase):
    cpp_name = "NNUtilsTest"

    def test_ClipGradNorm(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClipGradNorm")

    def test_ClipGradNormErrorIfNonfinite(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClipGradNormErrorIfNonfinite")

    def test_ClipGradValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClipGradValue")

    def test_ConvertParameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConvertParameters")

    def test_PackSequence(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PackSequence")

    def test_PackPaddedSequence(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PackPaddedSequence")

    def test_PadSequence(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PadSequence")


class TestPackedSequenceTest(TestCase):
    cpp_name = "PackedSequenceTest"

    def test_WrongOrder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "WrongOrder")

    def test_TotalLength(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TotalLength")

    def test_To(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "To")


class TestOptimTest(TestCase):
    cpp_name = "OptimTest"

    def test_OptimizerAccessors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OptimizerAccessors")

    def test_OldInterface(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OldInterface")

    def test_XORConvergence_SGD(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_SGD")

    def test_XORConvergence_LBFGS(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_LBFGS")

    def test_XORConvergence_Adagrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_Adagrad")

    def test_XORConvergence_RMSprop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_RMSprop")

    def test_XORConvergence_RMSpropWithMomentum(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_RMSpropWithMomentum")

    def test_XORConvergence_Adam(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_Adam")

    def test_XORConvergence_AdamWithAmsgrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_AdamWithAmsgrad")

    def test_ProducesPyTorchValues_Adam(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_Adam")

    def test_ProducesPyTorchValues_AdamWithWeightDecay(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_AdamWithWeightDecay"
        )

    def test_ProducesPyTorchValues_AdamWithWeightDecayAndAMSGrad(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ProducesPyTorchValues_AdamWithWeightDecayAndAMSGrad",
        )

    def test_XORConvergence_AdamW(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_AdamW")

    def test_XORConvergence_AdamWWithAmsgrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XORConvergence_AdamWWithAmsgrad")

    def test_ProducesPyTorchValues_AdamW(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_AdamW")

    def test_ProducesPyTorchValues_AdamWWithoutWeightDecay(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_AdamWWithoutWeightDecay"
        )

    def test_ProducesPyTorchValues_AdamWWithAMSGrad(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_AdamWWithAMSGrad"
        )

    def test_ProducesPyTorchValues_Adagrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_Adagrad")

    def test_ProducesPyTorchValues_AdagradWithWeightDecay(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_AdagradWithWeightDecay"
        )

    def test_ProducesPyTorchValues_AdagradWithWeightDecayAndLRDecay(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ProducesPyTorchValues_AdagradWithWeightDecayAndLRDecay",
        )

    def test_ProducesPyTorchValues_RMSprop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_RMSprop")

    def test_ProducesPyTorchValues_RMSpropWithWeightDecay(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_RMSpropWithWeightDecay"
        )

    def test_ProducesPyTorchValues_RMSpropWithWeightDecayAndCentered(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ProducesPyTorchValues_RMSpropWithWeightDecayAndCentered",
        )

    def test_ProducesPyTorchValues_RMSpropWithWeightDecayAndCenteredAndMomentum(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ProducesPyTorchValues_RMSpropWithWeightDecayAndCenteredAndMomentum",
        )

    def test_ProducesPyTorchValues_SGD(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_SGD")

    def test_ProducesPyTorchValues_SGDWithWeightDecay(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_SGDWithWeightDecay"
        )

    def test_ProducesPyTorchValues_SGDWithWeightDecayAndMomentum(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ProducesPyTorchValues_SGDWithWeightDecayAndMomentum",
        )

    def test_ProducesPyTorchValues_SGDWithWeightDecayAndNesterovMomentum(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "ProducesPyTorchValues_SGDWithWeightDecayAndNesterovMomentum",
        )

    def test_ProducesPyTorchValues_LBFGS(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_LBFGS")

    def test_ProducesPyTorchValues_LBFGS_with_line_search(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ProducesPyTorchValues_LBFGS_with_line_search"
        )

    def test_ZeroGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ZeroGrad")

    def test_ExternalVectorOfParameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExternalVectorOfParameters")

    def test_AddParameter_LBFGS(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AddParameter_LBFGS")

    def test_CheckLRChange_StepLR_Adam(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckLRChange_StepLR_Adam")


class TestOrderedDictTest(TestCase):
    cpp_name = "OrderedDictTest"

    def test_IsEmptyAfterDefaultConstruction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsEmptyAfterDefaultConstruction")

    def test_InsertAddsElementsWhenTheyAreYetNotPresent(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "InsertAddsElementsWhenTheyAreYetNotPresent"
        )

    def test_GetReturnsValuesWhenTheyArePresent(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetReturnsValuesWhenTheyArePresent")

    def test_GetThrowsWhenPassedKeysThatAreNotPresent(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "GetThrowsWhenPassedKeysThatAreNotPresent"
        )

    def test_CanInitializeFromList(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanInitializeFromList")

    def test_InsertThrowsWhenPassedElementsThatArePresent(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "InsertThrowsWhenPassedElementsThatArePresent"
        )

    def test_FrontReturnsTheFirstItem(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FrontReturnsTheFirstItem")

    def test_FrontThrowsWhenEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FrontThrowsWhenEmpty")

    def test_BackReturnsTheLastItem(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackReturnsTheLastItem")

    def test_BackThrowsWhenEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackThrowsWhenEmpty")

    def test_FindReturnsPointersToValuesWhenPresent(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "FindReturnsPointersToValuesWhenPresent"
        )

    def test_FindReturnsNullPointersWhenPasesdKeysThatAreNotPresent(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "FindReturnsNullPointersWhenPasesdKeysThatAreNotPresent",
        )

    def test_SubscriptOperatorThrowsWhenPassedKeysThatAreNotPresent(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "SubscriptOperatorThrowsWhenPassedKeysThatAreNotPresent",
        )

    def test_SubscriptOperatorReturnsItemsPositionallyWhenPassedIntegers(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "SubscriptOperatorReturnsItemsPositionallyWhenPassedIntegers",
        )

    def test_SubscriptOperatorsThrowswhenPassedKeysThatAreNotPresent(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "SubscriptOperatorsThrowswhenPassedKeysThatAreNotPresent",
        )

    def test_UpdateInsertsAllItemsFromAnotherOrderedDict(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UpdateInsertsAllItemsFromAnotherOrderedDict"
        )

    def test_UpdateAlsoChecksForDuplicates(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UpdateAlsoChecksForDuplicates")

    def test_CanIterateItems(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanIterateItems")

    def test_EraseWorks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EraseWorks")

    def test_ClearMakesTheDictEmpty(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ClearMakesTheDictEmpty")

    def test_CanCopyConstruct(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanCopyConstruct")

    def test_CanCopyAssign(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanCopyAssign")

    def test_CanMoveConstruct(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanMoveConstruct")

    def test_CanMoveAssign(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanMoveAssign")

    def test_CanInsertWithBraces(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanInsertWithBraces")

    def test_ErrorMessagesIncludeTheKeyDescription(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ErrorMessagesIncludeTheKeyDescription"
        )

    def test_KeysReturnsAllKeys(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "KeysReturnsAllKeys")

    def test_ValuesReturnsAllValues(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ValuesReturnsAllValues")

    def test_ItemsReturnsAllItems(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ItemsReturnsAllItems")


class TestRNNTest(TestCase):
    cpp_name = "RNNTest"

    def test_CheckOutputSizes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckOutputSizes")

    def test_CheckOutputSizesProj(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckOutputSizesProj")

    def test_CheckOutputValuesMatchPyTorch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CheckOutputValuesMatchPyTorch")

    def test_EndToEndLSTM(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EndToEndLSTM")

    def test_EndToEndLSTMProj(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EndToEndLSTMProj")

    def test_EndToEndGRU(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EndToEndGRU")

    def test_EndToEndRNNRelu(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EndToEndRNNRelu")

    def test_EndToEndRNNTanh(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EndToEndRNNTanh")

    def test_PrettyPrintRNNs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintRNNs")

    def test_BidirectionalFlattenParameters(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BidirectionalFlattenParameters")

    def test_BidirectionalGRUReverseForward(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BidirectionalGRUReverseForward")

    def test_BidirectionalLSTMReverseForward(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BidirectionalLSTMReverseForward")

    def test_UsePackedSequenceAsInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "UsePackedSequenceAsInput")


class TestSequentialTest(TestCase):
    cpp_name = "SequentialTest"

    def test_CanContainThings(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanContainThings")

    def test_ConstructsFromSharedPointer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromSharedPointer")

    def test_ConstructsFromConcreteType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromConcreteType")

    def test_ConstructsFromModuleHolder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsFromModuleHolder")

    def test_PushBackAddsAnElement(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PushBackAddsAnElement")

    def test_AccessWithAt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AccessWithAt")

    def test_AccessWithPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AccessWithPtr")

    def test_CallingForwardOnEmptySequentialIsDisallowed(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CallingForwardOnEmptySequentialIsDisallowed"
        )

    def test_CallingForwardChainsCorrectly(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CallingForwardChainsCorrectly")

    def test_CallingForwardWithTheWrongReturnTypeThrows(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "CallingForwardWithTheWrongReturnTypeThrows"
        )

    def test_TheReturnTypeOfForwardDefaultsToTensor(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TheReturnTypeOfForwardDefaultsToTensor"
        )

    def test_ForwardReturnsTheLastValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ForwardReturnsTheLastValue")

    def test_SanityCheckForHoldingStandardModules(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SanityCheckForHoldingStandardModules")

    def test_ExtendPushesModulesFromOtherSequential(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ExtendPushesModulesFromOtherSequential"
        )

    def test_HasReferenceSemantics(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "HasReferenceSemantics")

    def test_IsCloneable(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsCloneable")

    def test_RegistersElementsAsSubmodules(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegistersElementsAsSubmodules")

    def test_PrettyPrintSequential(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintSequential")

    def test_ModuleForwardMethodOptionalArg(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ModuleForwardMethodOptionalArg")


class TestTransformerTest(TestCase):
    cpp_name = "TransformerTest"

    def test_TransformerEncoderLayer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TransformerEncoderLayer")

    def test_TransformerDecoderLayer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TransformerDecoderLayer")

    def test_TransformerDecoderLayer_gelu(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TransformerDecoderLayer_gelu")

    def test_TransformerEncoder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TransformerEncoder")

    def test_PrettyPrintTransformerEncoderLayer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintTransformerEncoderLayer")

    def test_PrettyPrintTransformerEncoder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintTransformerEncoder")

    def test_PrettyPrintTransformerDecoderLayer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintTransformerDecoderLayer")

    def test_TransformerDecoder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TransformerDecoder")

    def test_PrettyPrintTransformerDecoder(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintTransformerDecoder")

    def test_Transformer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Transformer")

    def test_TransformerArgsCorrectness(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TransformerArgsCorrectness")


class TestSerializeTest(TestCase):
    cpp_name = "SerializeTest"

    def test_KeysFunc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "KeysFunc")

    def test_TryReadFunc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TryReadFunc")

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")

    def test_BasicToFile(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicToFile")

    def test_BasicViaFunc(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BasicViaFunc")

    def test_Resized(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Resized")

    def test_Sliced(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Sliced")

    def test_NonContiguous(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NonContiguous")

    def test_ErrorOnMissingKey(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ErrorOnMissingKey")

    def test_XOR(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "XOR")

    def test_Optim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Optim")

    def test_Optim_Adagrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Optim_Adagrad")

    def test_Optim_SGD(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Optim_SGD")

    def test_Optim_Adam(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Optim_Adam")

    def test_Optim_AdamW(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Optim_AdamW")

    def test_Optim_RMSprop(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Optim_RMSprop")

    def test_Optim_LBFGS(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Optim_LBFGS")

    def test_CanSerializeModulesWithIntermediateModulesWithoutParametersOrBuffers(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "CanSerializeModulesWithIntermediateModulesWithoutParametersOrBuffers",
        )

    def test_VectorOfTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "VectorOfTensors")

    def test_IValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IValue")

    def test_UnserializableSubmoduleIsSkippedWhenSavingModule(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "UnserializableSubmoduleIsSkippedWhenSavingModule",
        )

    def test_UnserializableSubmoduleIsIgnoredWhenLoadingModule(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "UnserializableSubmoduleIsIgnoredWhenLoadingModule",
        )


class TestSpecialTest(TestCase):
    cpp_name = "SpecialTest"

    def test_special(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "special")


class TestTestStatic(TestCase):
    cpp_name = "TestStatic"

    def test_AllOf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AllOf")

    def test_AnyOf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AnyOf")

    def test_EnableIfModule(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "EnableIfModule")

    def test_ReturnTypeOfForward(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReturnTypeOfForward")

    def test_Apply(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Apply")


class TestTensorTest(TestCase):
    cpp_name = "TensorTest"

    def test_ToDtype(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ToDtype")

    def test_ToTensorAndTensorAttributes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ToTensorAndTensorAttributes")

    def test_ToOptionsWithRequiresGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ToOptionsWithRequiresGrad")

    def test_ToDoesNotCopyWhenOptionsAreAllTheSame(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "ToDoesNotCopyWhenOptionsAreAllTheSame"
        )

    def test_AtTensorCtorScalar(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AtTensorCtorScalar")

    def test_AtTensorCtorSingleDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AtTensorCtorSingleDim")

    def test_AtTensorCastRealToComplex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AtTensorCastRealToComplex")

    def test_AtTensorCastComplexToRealErrorChecks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AtTensorCastComplexToRealErrorChecks")

    def test_TorchTensorCtorScalarIntegralType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorScalarIntegralType")

    def test_TorchTensorCtorScalarFloatingType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorScalarFloatingType")

    def test_TorchTensorCtorScalarBoolType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorScalarBoolType")

    def test_TorchTensorCtorSingleDimIntegralType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorSingleDimIntegralType")

    def test_TorchTensorCtorSingleDimFloatingType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorSingleDimFloatingType")

    def test_TorchTensorCtorSingleDimBoolType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorSingleDimBoolType")

    def test_TorchTensorCtorMultiDimIntegralType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorMultiDimIntegralType")

    def test_TorchTensorCtorMultiDimFloatingType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorMultiDimFloatingType")

    def test_TorchTensorCtorMultiDimBoolType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorMultiDimBoolType")

    def test_TorchTensorCtorMultiDimWithOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorMultiDimWithOptions")

    def test_TorchTensorCtorMultiDimErrorChecks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorMultiDimErrorChecks")

    def test_TorchTensorCastRealToComplex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCastRealToComplex")

    def test_TorchTensorCastComplexToRealErrorChecks(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TorchTensorCastComplexToRealErrorChecks"
        )

    def test_TorchTensorCtorZeroSizedDim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorZeroSizedDim")

    def test_TorchTensorCtorWithoutSpecifyingDtype(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TorchTensorCtorWithoutSpecifyingDtype"
        )

    def test_TorchTensorCtorWithNonDtypeOptions(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TorchTensorCtorWithNonDtypeOptions")

    def test_Arange(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Arange")

    def test_PrettyPrintTensorDataContainer(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PrettyPrintTensorDataContainer")

    def test_TensorDataContainerCallingAccessorOfWrongType(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TensorDataContainerCallingAccessorOfWrongType"
        )

    def test_FromBlob(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FromBlob")

    def test_FromBlobUsesDeleter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FromBlobUsesDeleter")

    def test_FromBlobWithStrides(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FromBlobWithStrides")

    def test_Item(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Item")

    def test_DataPtr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DataPtr")

    def test_Data(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Data")

    def test_BackwardAndGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardAndGrad")

    def test_BackwardCreatesOnesGrad(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardCreatesOnesGrad")

    def test_BackwardNonScalarOutputs(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "BackwardNonScalarOutputs")

    def test_IsLeaf(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "IsLeaf")

    def test_OutputNr(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OutputNr")

    def test_Version(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Version")

    def test_Detach(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Detach")

    def test_DetachInplace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DetachInplace")

    def test_SetData(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetData")

    def test_RequiresGradInplace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RequiresGradInplace")

    def test_StdDimension(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "StdDimension")

    def test_ReshapeAlias(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ReshapeAlias")


class TestTensorIndexingTest(TestCase):
    cpp_name = "TensorIndexingTest"

    def test_Slice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Slice")

    def test_TensorIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TensorIndex")

    def test_TestNoIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNoIndices")

    def test_TestAdvancedIndexingWithListOfTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestAdvancedIndexingWithListOfTensor")

    def test_TestSingleInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSingleInt")

    def test_TestMultipleInt(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMultipleInt")

    def test_TestNone(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNone")

    def test_TestStep(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestStep")

    def test_TestStepAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestStepAssignment")

    def test_TestBoolIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBoolIndices")

    def test_TestBoolIndicesAccumulate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBoolIndicesAccumulate")

    def test_TestMultipleBoolIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMultipleBoolIndices")

    def test_TestByteMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestByteMask")

    def test_TestByteMaskAccumulate(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestByteMaskAccumulate")

    def test_TestMultipleByteMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestMultipleByteMask")

    def test_TestByteMask2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestByteMask2d")

    def test_TestIntIndices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIntIndices")

    def test_TestIntIndices2d(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIntIndices2d")

    def test_TestIntIndicesBroadcast(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIntIndicesBroadcast")

    def test_TestEmptyIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEmptyIndex")

    def test_TestEmptyNdimIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEmptyNdimIndex")

    def test_TestEmptyNdimIndexBool(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEmptyNdimIndexBool")

    def test_TestEmptySlice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEmptySlice")

    def test_TestIndexGetitemCopyBoolsSlices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIndexGetitemCopyBoolsSlices")

    def test_TestIndexSetitemBoolsSlices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIndexSetitemBoolsSlices")

    def test_TestIndexScalarWithBoolMask(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIndexScalarWithBoolMask")

    def test_TestSetitemExpansionError(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSetitemExpansionError")

    def test_TestGetitemScalars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestGetitemScalars")

    def test_TestSetitemScalars(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSetitemScalars")

    def test_TestBasicAdvancedCombined(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBasicAdvancedCombined")

    def test_TestIntAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIntAssignment")

    def test_TestByteTensorAssignment(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestByteTensorAssignment")

    def test_TestVariableSlicing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestVariableSlicing")

    def test_TestEllipsisTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEllipsisTensor")

    def test_TestOutOfBoundIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestOutOfBoundIndex")

    def test_TestZeroDimIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestZeroDimIndex")


class TestNumpyTests(TestCase):
    cpp_name = "NumpyTests"

    def test_TestNoneIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestNoneIndex")

    def test_TestEmptyFancyIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEmptyFancyIndex")

    def test_TestEllipsisIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEllipsisIndex")

    def test_TestSingleIntIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSingleIntIndex")

    def test_TestSingleBoolIndex(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSingleBoolIndex")

    def test_TestBooleanShapeMismatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBooleanShapeMismatch")

    def test_TestBooleanIndexingOnedim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBooleanIndexingOnedim")

    def test_TestBooleanAssignmentValueMismatch(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBooleanAssignmentValueMismatch")

    def test_TestBooleanIndexingTwodim(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBooleanIndexingTwodim")

    def test_TestBooleanIndexingWeirdness(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBooleanIndexingWeirdness")

    def test_TestBooleanIndexingWeirdnessTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBooleanIndexingWeirdnessTensors")

    def test_TestBooleanIndexingAlldims(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBooleanIndexingAlldims")

    def test_TestBooleanListIndexing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBooleanListIndexing")

    def test_TestEverythingReturnsViews(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestEverythingReturnsViews")

    def test_TestBroaderrorsIndexing(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBroaderrorsIndexing")

    def test_TestTrivialFancyOutOfBounds(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestTrivialFancyOutOfBounds")

    def test_TestIndexIsLarger(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestIndexIsLarger")

    def test_TestBroadcastSubspace(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestBroadcastSubspace")


class TestTensorOptionsTest(TestCase):
    cpp_name = "TensorOptionsTest"

    def test_DefaultsToTheRightValues(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DefaultsToTheRightValues")

    def test_UtilityFunctionsReturnTheRightTensorOptions(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "UtilityFunctionsReturnTheRightTensorOptions"
        )

    def test_ConstructsWellFromCPUTypes(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsWellFromCPUTypes")

    def test_ConstructsWellFromCPUTensors(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsWellFromCPUTensors")

    def test_ConstructsWellFromVariables(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructsWellFromVariables")


class TestDeviceTest(TestCase):
    cpp_name = "DeviceTest"

    def test_ParsesCorrectlyFromString(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ParsesCorrectlyFromString")


class TestDefaultDtypeTest(TestCase):
    cpp_name = "DefaultDtypeTest"

    def test_CanSetAndGetDefaultDtype(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CanSetAndGetDefaultDtype")

    def test_NewTensorOptionsHasCorrectDefault(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NewTensorOptionsHasCorrectDefault")

    def test_NewTensorsHaveCorrectDefaultDtype(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NewTensorsHaveCorrectDefaultDtype")


class TestTorchIncludeTest(TestCase):
    cpp_name = "TorchIncludeTest"

    def test_GetSetNumThreads(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GetSetNumThreads")


class TestInferenceModeTest(TestCase):
    cpp_name = "InferenceModeTest"

    def test_TestTLSState(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestTLSState")

    def test_TestInferenceTensorCreation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestInferenceTensorCreation")

    def test_TestExistingAutogradSession(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestExistingAutogradSession")

    def test_TestInferenceTensorInInferenceModeFunctionalOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestInferenceTensorInInferenceModeFunctionalOp"
        )

    def test_TestInferenceTensorInInferenceModeInplaceOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestInferenceTensorInInferenceModeInplaceOp"
        )

    def test_TestInferenceTensorInInferenceModeViewOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestInferenceTensorInInferenceModeViewOp"
        )

    def test_TestInferenceTensorInNormalModeFunctionalOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestInferenceTensorInNormalModeFunctionalOp"
        )

    def test_TestInferenceTensorInNormalModeInplaceOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestInferenceTensorInNormalModeInplaceOp"
        )

    def test_TestInferenceTensorInNormalModeViewOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestInferenceTensorInNormalModeViewOp"
        )

    def test_TestNormalTensorInplaceOutputInInferenceMode(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestNormalTensorInplaceOutputInInferenceMode"
        )

    def test_TestNormalTensorInplaceOutputInNormalMode(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestNormalTensorInplaceOutputInNormalMode"
        )

    def test_TestNormalTensorViewOutputInInferenceMode(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestNormalTensorViewOutputInInferenceMode"
        )

    def test_TestNormalTensorViewOutputInNormalMode(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestNormalTensorViewOutputInNormalMode"
        )

    def test_TestMixInferenceAndNormalTensorFunctionalOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestMixInferenceAndNormalTensorFunctionalOp"
        )

    def test_TestMixInferenceAndNormalTensorInplaceOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestMixInferenceAndNormalTensorInplaceOp"
        )

    def test_TestMixInferenceAndNormalTensorViewOp(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestMixInferenceAndNormalTensorViewOp"
        )

    def test_TestHandleDirectViewOnRebase(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestHandleDirectViewOnRebase")

    def test_TestHandleInDirectViewOnRebase(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestHandleInDirectViewOnRebase")

    def test_TestCreationMetaPropagation(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCreationMetaPropagation")

    def test_TestCreationMetaPropagationInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCreationMetaPropagationInput")

    def test_TestInplaceCopyOnInferenceTensor(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestInplaceCopyOnInferenceTensor")

    def test_TestSetRequiresGradInNormalMode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSetRequiresGradInNormalMode")

    def test_TestAccessVersionCounter(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestAccessVersionCounter")

    def test_TestInplaceUpdateInferenceTensorWithNormalTensor(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "TestInplaceUpdateInferenceTensorWithNormalTensor",
        )

    def test_TestComplexViewInInferenceMode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestComplexViewInInferenceMode")

    def test_TestComplexViewInNormalMode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestComplexViewInNormalMode")

    def test_TestCustomFunction(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestCustomFunction")

    def test_TestLegacyAutoNonVariableTypeModeWarning(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "TestLegacyAutoNonVariableTypeModeWarning"
        )


class TestGradModeTest(TestCase):
    cpp_name = "GradModeTest"

    def test_TestRequiresGradFunctionalOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRequiresGradFunctionalOp")

    def test_TestRequiresGradInplaceOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRequiresGradInplaceOp")

    def test_TestRequiresGradViewOp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRequiresGradViewOp")

    def test_TestRequiresGradViewOpExiting(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestRequiresGradViewOpExiting")


class TestOperationTest(TestCase):
    cpp_name = "OperationTest"

    def test_Lerp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Lerp")

    def test_Cross(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Cross")

    def test_Linear_out(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Linear_out")


if __name__ == "__main__":
    run_tests()

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly -skip-spirv-validation

// CHECK: type: float
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 9.000000
// CHECK-NEXT: 12.000000

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 4.0], stride=4, count=256):name=inputBuffer
StructuredBuffer<float> inputBuffer;

//TEST_INPUT:ubuffer(stride=4, count=256):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typealias CoopMatType = CoopMat<float, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

static let stride = 16;
//static let matrixLayout = CoopMatMatrixLayout::RowMajor;
static const CoopMatMatrixLayout matrixLayout = CoopMatMatrixLayout::RowMajor;

void processCoopMat(CoopMatType mat)
{
    // XXX: hmmm, some error when matrixLAyout is static let
    (mat * 3.0).store(outputBuffer, 0, stride, matrixLayout);
}

[numthreads(32, 1, 1)]
void computeMain()
{
    let mat = CoopMatType.load(inputBuffer, 0, stride, matrixLayout);
    processCoopMat(mat);
}

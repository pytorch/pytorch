//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry computeMain -stage compute -profile sm_5_0
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target glsl -emit-spirv-directly -entry computeMain -stage compute -profile sm_5_0 -ignore-capabilities -skip-spirv-validation

// Test that we diagnose simplified capabilities
// CHECK_IGNORE_CAPS-NOT: error 36104
// CHECK-NOT: sm_4_0
// CHECK-NOT: sm_5_0
// CHECK-NOT: sm_5_1
// CHECK: error 36104: 'processDataBad' uses undeclared capability 'sm_6_0'
// CHECK: capability7.slang(28): note: see using of 'processDataBadNested'
// CHECK: capability7.slang(20): note: see definition of 'processDataBadNested'


[require(glsl_hlsl_metal_spirv)]
void processDataGood()
{
}

[require(hlsl, sm_6_0)]
void processDataBadNested()
{
    AllMemoryBarrier();
}

[require(hlsl)]
void processDataBad()
{
    processDataBadNested();
}

void myNestedNestedSafeCall()
{
    processDataGood();
}

void myNestedNestedBadCall()
{
    processDataGood();
    processDataBad();
}

void myNestedCall()
{
    myNestedNestedSafeCall();
    myNestedNestedBadCall();
}

[numthreads(1,1,1)]
void computeMain()
{
    myNestedCall();
}

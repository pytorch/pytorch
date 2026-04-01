//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK_HLSL): -allow-glsl -stage compute -entry computeMain -target hlsl -DTARGET_HLSL

// not testing cuda due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_CUDA): -allow-glsl -stage compute -entry computeMain -target cuda -DTARGET_CUDA 
// not testing cpp due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_CPP):  -allow-glsl -stage compute -entry computeMain -target cpp -DTARGET_CPP

//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-wgpu -compute -entry computeMain -allow-glsl -xslang -DWGPU
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-metal -compute -entry computeMain -allow-glsl

#version 430

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
buffer MyBlockName2 
{
    uint data[];
} outputBuffer;

layout(local_size_x = 32) in;

shared uint shareMem;

[[ForceInline]]
void _barrier()
{
#if !defined(WGPU)
    subgroupBarrier();
#else
    GroupMemoryBarrier();
#endif
}

[[ForceInline]]
void _memoryBarrier()
{
#if !defined(WGPU)
    subgroupMemoryBarrier();
#else
    GroupMemoryBarrier();
#endif
}

[[ForceInline]]
void _memoryBarrierShared()
{
#if !defined(WGPU)
    subgroupMemoryBarrierShared();
#else
    GroupMemoryBarrier();
#endif
}

[[ForceInline]]
void _memoryBarrierBuffer()
{
#if !defined(WGPU)
    subgroupMemoryBarrierBuffer();
#else
    GroupMemoryBarrier();
#endif
}

void computeMain()
{
    // TODO: no test for image memory was done -- subgroupMemoryBarrierImage();
    // tests are seperate since concurrency testing

    shareMem = 100;
    _memoryBarrierShared();
    outputBuffer.data[0] = 1;
    _barrier();
    outputBuffer.data[0] = 2;
    _barrier();

    outputBuffer.data[1] = 1;
    _memoryBarrier();
    outputBuffer.data[1] = 2;
    _barrier();

    outputBuffer.data[2] = 1;
    _memoryBarrierBuffer();
    outputBuffer.data[2] = 2;
    _barrier();

    shareMem = 2;
    _memoryBarrierShared();
    outputBuffer.data[3] = shareMem;
    _barrier();

    if (subgroupElect()) {
        outputBuffer.data[4] = gl_GlobalInvocationID.x + 2;
    }

    // CHECK_GLSL: void main(
    // CHECK_SPV: OpEntryPoint
    // CHECK_HLSL: void computeMain(
    // CHECK_CUDA: void computeMain(
    // CHECK_CPP: void _computeMain(

    // BUF: 2
    // BUF-NEXT: 2
    // BUF-NEXT: 2
    // BUF-NEXT: 2
    // BUF-NEXT: 2
}

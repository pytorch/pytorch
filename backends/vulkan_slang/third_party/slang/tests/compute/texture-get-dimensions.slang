//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12  -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile cs_6_0 -use-dxil -shaderobj
// TODO(JS): Doesn't work on vk currently
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
// TODO(JS): Doesn't work on CUDA as there aren't any CUDA functions to get dimensions.
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute  -shaderobj
// Not supported in WGSL: 1D array texture not supported in WGSL
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -wgpu

//TEST_INPUT: Texture1D(size=4, content = one):name t1D
Texture1D<float> t1D;
//TEST_INPUT: Texture2D(size=8, content = one):name t2D
Texture2D<float> t2D;
//TEST_INPUT: Texture3D(size=2, content = one):name t3D
Texture3D<float> t3D;
//TEST_INPUT: TextureCube(size=16, content = one):name tCube
TextureCube<float> tCube;

//TEST_INPUT: Texture1D(size=2, content = one, arrayLength=2):name t1DArray
Texture1DArray<float> t1DArray;
//TEST_INPUT: Texture2D(size=4, content = one, arrayLength=3):name t2DArray
Texture2DArray<float> t2DArray;
//TEST_INPUT: TextureCube(size=16, content = one, arrayLength=1):name tCubeArray
TextureCubeArray<float> tCubeArray;

//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;

//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    
    uint mipWidth = 0;
    uint mipHeight = 0;
    uint mipDepth = 0;
    uint mipCount = 1;
    uint elementCount = 0;
    
    uint width = 0;
    uint height = 0;
    uint depth = 0;
    
    switch (idx)
    {
        case 0:
        {
            t1D.GetDimensions(width);
            t1D.GetDimensions(0, mipWidth, mipCount);
            break;
        }
        case 1:
        {
            t2D.GetDimensions(width, height);
            t2D.GetDimensions(0, mipWidth, mipHeight, mipCount);
            break;
        }
        case 2:
        {
            t3D.GetDimensions(width, height, depth);
            t3D.GetDimensions(0, mipWidth, mipHeight, mipDepth, mipCount);
            break;
        }
        case 3:
        {
            tCube.GetDimensions(width, height);
            tCube.GetDimensions(0, mipWidth, mipHeight, mipCount);
            break;
        }
        case 4:
        {
           t1DArray.GetDimensions(width, elementCount);
           t1DArray.GetDimensions(0, mipWidth, elementCount, mipCount);
           break;
        }
        case 5:
        {
           t2DArray.GetDimensions(width, height, elementCount);
           t2DArray.GetDimensions(0, mipWidth, mipHeight, elementCount, mipCount);
           break;
        }
        case 6:
        {
           tCubeArray.GetDimensions(width, height, elementCount);
           tCubeArray.GetDimensions(0, mipWidth, mipHeight, elementCount, mipCount);
           break;
        }
    }
   
    width = (mipWidth != width) ? ~0 : width;
    height = (mipHeight != height) ? ~0 : height;
    depth = (mipDepth != depth) ? ~0 : depth;
   
    // Only depth or element count can be set
    if (depth == 0 && elementCount > 0)
    {
        depth = elementCount;
    }
   
    uint val = ((0xff & width) << 24) | ((0xff & height) << 16) | ((0xff & depth) << 8) | (mipCount);
   
    outputBuffer[idx] = val;
}

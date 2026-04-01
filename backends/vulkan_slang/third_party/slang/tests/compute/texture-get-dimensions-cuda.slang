//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

// Only testing the dimensions available in CUDA via txq:
// - txq.width.b32
// - txq.height.b32
// - txq.depth.b32

//TEST_INPUT: Texture1D(size=4, content = one):name cudaT1D
Texture1D<float> cudaT1D;
//TEST_INPUT: Texture2D(size=8, content = one):name cudaT2D
Texture2D<float> cudaT2D;
//TEST_INPUT: Texture3D(size=2, content = one):name cudaT3D
Texture3D<float> cudaT3D;
//TEST_INPUT: TextureCube(size=16, content = one):name cudaTCube
TextureCube<float> cudaTCube;
//TEST_INPUT: Texture2D(size=4, content = one, arrayLength=3):name cudaT2DArray
Texture2DArray<float> cudaT2DArray;
//TEST_INPUT: TextureCube(size=16, content = one, arrayLength=1):name cudaTCubeArray
TextureCubeArray<float> cudaTCubeArray;

//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0], stride=4):out,name cudaOutputBuffer
RWStructuredBuffer<uint> cudaOutputBuffer;

[numthreads(7, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    
    uint width = 0;
    uint height = 0;
    uint depth = 0;
    
    switch (idx)
    {
        case 0:
        {
            // Test 1D texture width
            cudaT1D.GetDimensions(width);
            cudaOutputBuffer[idx] = width;
            break;
        }
        case 1:
        {
            // Test 2D texture width and height
            cudaT2D.GetDimensions(width, height);
            cudaOutputBuffer[idx] = ((0xff & width) << 8) | (0xff & height);
            break;
        }
        case 2:
        {
            // Test 3D texture width, height and depth
            cudaT3D.GetDimensions(width, height, depth);
            cudaOutputBuffer[idx] = ((0xff & width) << 16) | ((0xff & height) << 8) | (0xff & depth);
            break;
        }
        case 3:
        {
            // Test Cube texture width and height
            cudaTCube.GetDimensions(width, height);
            cudaOutputBuffer[idx] = ((0xff & width) << 8) | (0xff & height);
            break;
        }
        case 4:
        {
            // Test 2D array texture width and height
            // Note: We don't test array size since txq.array_size is not implemented
            cudaT2DArray.GetDimensions(width, height, depth);
            cudaOutputBuffer[idx] = ((0xff & width) << 8) | (0xff & height);
            break;
        }
        case 5:
        {
            // Test Cube array texture width and height
            // Note: We don't test array size since txq.array_size is not implemented
            cudaTCubeArray.GetDimensions(width, height, depth);
            cudaOutputBuffer[idx] = ((0xff & width) << 8) | (0xff & height);
            break;
        }
        case 6:
        {
            // Reserved for future use
            cudaOutputBuffer[idx] = 0;
            break;
        }
    }
} 
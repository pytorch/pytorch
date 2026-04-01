//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;

[Differentiable]
float _clamp(float x, float min, float max)
{
    return clamp(x, min, max);
}

[Differentiable]
float3 _clamp3(float3 x, float3 min, float3 max)
{
    return clamp(x, min, max);
}

[Differentiable]
float _clamp_equiv(float x, float _min, float _max)
{
    return max(_min, min(_max, x));
}

[Differentiable]
float3 _clamp_equiv(float3 x, float3 _min, float3 _max)
{
    return max(_min, min(_max, x));
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // x in between max and min
    {
        dpfloat dpx = dpfloat(2.0, 0.1);
        dpfloat dpmax = dpfloat(3.0, 0.2);
        dpfloat dpmin = dpfloat(1.0, 0.3);

        dpfloat res = fwd_diff(_clamp)(dpx, dpmin, dpmax);
        outputBuffer[0] = res.d; // Expected: 0.1
    }

    // x less than min
    {
        dpfloat dpx = dpfloat(0.5, 0.1);
        dpfloat dpmax = dpfloat(3.0, 0.2);
        dpfloat dpmin = dpfloat(1.0, 0.3);

        dpfloat res = fwd_diff(_clamp)(dpx, dpmin, dpmax);
        outputBuffer[1] = res.d; // Expected: 0.3
    }

    // x greater than max
    {
        dpfloat dpx = dpfloat(4.0, 0.1);
        dpfloat dpmax = dpfloat(3.0, 0.2);
        dpfloat dpmin = dpfloat(1.0, 0.3);

        dpfloat res = fwd_diff(_clamp)(dpx, dpmin, dpmax);
        outputBuffer[2] = res.d; // Expected: 0.2
    }

    // float3 version with one in between, one below min and one above max.
    {
        dpfloat3 dpx = dpfloat3(float3(2.0, 0.5, 4.0), float3(0.1, 0.1, 0.1));
        dpfloat3 dpmax = dpfloat3(float3(3.0, 3.0, 3.0), float3(0.2, 0.2, 0.2));
        dpfloat3 dpmin = dpfloat3(float3(1.0, 1.0, 1.0), float3(0.3, 0.3, 0.3));

        dpfloat3 res = fwd_diff(_clamp3)(dpx, dpmin, dpmax);
        outputBuffer[3] = res.d.x; // Expected: 0.1
        outputBuffer[4] = res.d.y; // Expected: 0.3
        outputBuffer[5] = res.d.z; // Expected: 0.2
    }

    // Equivalent to the first test, but with a different implementation of clamp
    {
        dpfloat dpx = dpfloat(2.0, 0.1);
        dpfloat dpmax = dpfloat(3.0, 0.2);
        dpfloat dpmin = dpfloat(1.0, 0.3);

        dpfloat res = fwd_diff(_clamp_equiv)(dpx, dpmin, dpmax);
        outputBuffer[6] = res.d; // Expected: 0.1
    }

    // Equivalent to the second test, but with a different implementation of clamp
    {
        dpfloat dpx = dpfloat(0.5, 0.1);
        dpfloat dpmax = dpfloat(3.0, 0.2);
        dpfloat dpmin = dpfloat(1.0, 0.3);

        dpfloat res = fwd_diff(_clamp_equiv)(dpx, dpmin, dpmax);
        outputBuffer[7] = res.d; // Expected: 0.3
    }

    // Equivalent to the third test, but with a different implementation of clamp
    {
        dpfloat dpx = dpfloat(4.0, 0.1);
        dpfloat dpmax = dpfloat(3.0, 0.2);
        dpfloat dpmin = dpfloat(1.0, 0.3);

        dpfloat res = fwd_diff(_clamp_equiv)(dpx, dpmin, dpmax);
        outputBuffer[8] = res.d; // Expected: 0.2
    }

    // Equivalent to the fourth test, but with a different implementation of clamp
    {
        dpfloat3 dpx = dpfloat3(float3(2.0, 0.5, 4.0), float3(0.1, 0.1, 0.1));
        dpfloat3 dpmax = dpfloat3(float3(3.0, 3.0, 3.0), float3(0.2, 0.2, 0.2));
        dpfloat3 dpmin = dpfloat3(float3(1.0, 1.0, 1.0), float3(0.3, 0.3, 0.3));

        dpfloat3 res = fwd_diff(_clamp_equiv)(dpx, dpmin, dpmax);
        outputBuffer[9] = res.d.x; // Expected: 0.1
        outputBuffer[10] = res.d.y; // Expected: 0.3
        outputBuffer[11] = res.d.z; // Expected: 0.2
    }

    // Reverse-mode tests.
    
    // x in between max and min
    {
        dpfloat dpx = dpfloat(2.0, 0.0);
        dpfloat dpmax = dpfloat(3.0, 0.0);
        dpfloat dpmin = dpfloat(1.0, 0.0);

        bwd_diff(_clamp)(dpx, dpmin, dpmax, 1.0);
        
        outputBuffer[12] = dpx.d; // Expected: 1.0
        outputBuffer[13] = dpmin.d; // Expected: 0.0
        outputBuffer[14] = dpmax.d; // Expected: 0.0
    }

    // x less than min
    {
        dpfloat dpx = dpfloat(0.5, 0.0);
        dpfloat dpmax = dpfloat(3.0, 0.0);
        dpfloat dpmin = dpfloat(1.0, 0.0);

        bwd_diff(_clamp)(dpx, dpmin, dpmax, 1.0);
        
        outputBuffer[15] = dpx.d; // Expected: 0.0
        outputBuffer[16] = dpmin.d; // Expected: 1.0
        outputBuffer[17] = dpmax.d; // Expected: 0.0
    }

    // x greater than max
    {
        dpfloat dpx = dpfloat(4.0, 0.0);
        dpfloat dpmax = dpfloat(3.0, 0.0);
        dpfloat dpmin = dpfloat(1.0, 0.0);

        bwd_diff(_clamp)(dpx, dpmin, dpmax, 1.0);
        
        outputBuffer[18] = dpx.d; // Expected: 0.0
        outputBuffer[19] = dpmin.d; // Expected: 0.0
        outputBuffer[20] = dpmax.d; // Expected: 1.0
    }

    // float3 version with one in between, one below min and one above max.
    {
        dpfloat3 dpx = dpfloat3(float3(2.0, 0.5, 4.0), float3(0.0, 0.0, 0.0));
        dpfloat3 dpmax = dpfloat3(float3(3.0, 3.0, 3.0), float3(0.0, 0.0, 0.0));
        dpfloat3 dpmin = dpfloat3(float3(1.0, 1.0, 1.0), float3(0.0, 0.0, 0.0));

        bwd_diff(_clamp3)(dpx, dpmin, dpmax, float3(0.1, 0.2, 0.3));
        
        outputBuffer[21] = dpx.d.x; // Expected: 0.1
        outputBuffer[22] = dpx.d.y; // Expected: 0.0
        outputBuffer[23] = dpx.d.z; // Expected: 0.0
        outputBuffer[24] = dpmin.d.x; // Expected: 0.0
        outputBuffer[25] = dpmin.d.y; // Expected: 0.2
        outputBuffer[26] = dpmin.d.z; // Expected: 0.0
        outputBuffer[27] = dpmax.d.x; // Expected: 0.0
        outputBuffer[28] = dpmax.d.y; // Expected: 0.0
        outputBuffer[29] = dpmax.d.z; // Expected: 0.3
    }

    // New tests: Forward-mode tests for derivative propagation at the edges with clamp(x, 0, 1)
    {
        // Lower edge: x exactly = 0
        dpfloat dpx = dpfloat(0.0, 0.4);
        dpfloat dpmin = dpfloat(0.0, 0.8);
        dpfloat dpmax = dpfloat(1.0, 0.5);
        dpfloat res = fwd_diff(_clamp)(dpx, dpmin, dpmax);
        outputBuffer[30] = res.d; // Expected: 0.4 (propagated from x)
    }

    {
        // Upper edge: x exactly = 1
        dpfloat dpx = dpfloat(1.0, 0.7);
        dpfloat dpmin = dpfloat(0.0, 0.8);
        dpfloat dpmax = dpfloat(1.0, 0.9);
        dpfloat res = fwd_diff(_clamp)(dpx, dpmin, dpmax);
        outputBuffer[31] = res.d; // Expected: 0.7 (propagated from x)
    }

    // Reverse-mode tests for derivative propagation at the edges with clamp(x, 0, 1)
    {
        // Lower edge: x exactly = 0
        dpfloat dpx = dpfloat(0.0, 0.0);
        dpfloat dpmin = dpfloat(0.0, 0.0);
        dpfloat dpmax = dpfloat(1.0, 0.0);
        bwd_diff(_clamp)(dpx, dpmin, dpmax, 1.0);
        outputBuffer[32] = dpx.d;   // Expected: 1.0 (propagated from x)
        outputBuffer[33] = dpmin.d; // Expected: 0.0
        outputBuffer[34] = dpmax.d; // Expected: 0.0
    }

    {
        // Upper edge: x exactly = 1
        dpfloat dpx = dpfloat(1.0, 0.0);
        dpfloat dpmin = dpfloat(0.0, 0.0);
        dpfloat dpmax = dpfloat(1.0, 0.0);
        bwd_diff(_clamp)(dpx, dpmin, dpmax, 1.0);
        outputBuffer[35] = dpx.d;   // Expected: 1.0 (propagated from x)
        outputBuffer[36] = dpmin.d; // Expected: 0.0
        outputBuffer[37] = dpmax.d; // Expected: 0.0
    }
}

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -output-using-type -shaderobj
//TEST:SIMPLE(filecheck=CHK_REPORT):-target hlsl -stage compute -entry computeMain -report-checkpoint-intermediates

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

// A variety of tests to check for loop exit value inference.
// For all of these loops, we expect our inference pass to be able to
// infer the loop exit value correctly.
//
// Further, if the optimization pass runs successfully, then there should
// be absolutely no context stored for any of these tests.
//

// CHK_REPORT: (0): note: no checkpoint contexts to report

[Differentiable]
float test_simple(float y)
{
    float t = y;

    for (int i = 0; i < 3; i++)
    {
        t = t * (i + 1);
    }

    return t;
}

[Differentiable]
float test_strided(float y)
{
    float t = y;

    for (int i = 0; i < 5; i+=2)
    {
        t = t * (i + 1);
    }

    return t;
}

[Differentiable]
float test_offset(float y)
{
    float t = y;

    for (int i = 2; i < 5; i+=2)
    {
        t = t * (i + 1);
    }

    return t;
}

[Differentiable]
float test_negative_stride(float y)
{
    float t = y;

    for (int i = 7; i >= 1; i-=2)
    {
        t = t * (i + 1);
    }

    return t;
}

[Differentiable]
float test_nested(float y)
{
    float t = y;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            t = t * (i + 4 * j + 1);
        }
    }

    return t;
}

[Differentiable]
float test_nested_with_offset(float y)
{
    float t = y;

    for (int i = -3; i < 3; i++)
    {
        for (int j = -3; j < 3; j++)
        {
            t = t * ((abs(i) % 2) + (abs(j) % 2) + 1);
        }
    }

    return t;
}

[Differentiable]
float test_nested_with_conditions(float y)
{
    float t = y;

    for (int i = 0; i < 3; i++)
    {
        if (i % 2 == 0)
        {
            for (int j = 0; j < 3; j++)
            {
                if (j % 2 == 0)
                {
                    t = t * (i + 4 * j + 1);
                }
            }
        }
    }

    return t;
}

[Differentiable]
float test_with_continue(float y)
{
    float t = y;

    for (int i = 0; i < 5; i++)
    {
        if (i % 2 == 0)
        {
            continue;
        }

        t = t * (i + 1);
    }

    return t;
}

[Differentiable]
float test_nested_with_continue(float y)
{
    float t = y;

    for (int i = 0; i < 3; i++)
    {
        if (i % 2 == 0)
            continue;
    
        for (int j = 0; j < 3; j++)
        {
            if (j % 2 == 0)
                continue;

            if (j == 0)
                continue;

            t = t * (i + 4 * j + 1);
        }
    }

    return t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[0] = 0.0f; // CHECK: 0.000000

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_simple)(dpa, 1.0f);
        outputBuffer[1] = dpa.d; // CHECK-NEXT: 6.000000
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_strided)(dpa, 1.0f);
        outputBuffer[2] = dpa.d; // CHECK-NEXT: 15.000000
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_offset)(dpa, 1.0f);
        outputBuffer[3] = dpa.d; // CHECK-NEXT: 15.000000
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_negative_stride)(dpa, 1.0f);
        outputBuffer[4] = dpa.d; // CHECK-NEXT: 384.000000
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_nested)(dpa, 1.0f);
        outputBuffer[5] = dpa.d;  // CHECK-NEXT: 1247400.000000
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_nested_with_offset)(dpa, 1.0f);
        outputBuffer[6] = dpa.d; // CHECK-NEXT: 5159780352.000000
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_nested_with_conditions)(dpa, 1.0f);
        outputBuffer[7] = dpa.d; // CHECK-NEXT: 297.000000
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_with_continue)(dpa, 1.0f);
        outputBuffer[8] = dpa.d; // CHECK-NEXT: 8.000000
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_nested_with_continue)(dpa, 1.0f);
        outputBuffer[9] = dpa.d; // CHECK-NEXT: 6.000000
    }
}

//CHK-NOT: note
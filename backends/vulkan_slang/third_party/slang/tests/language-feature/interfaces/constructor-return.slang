//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Executable test to check the correctness of return values from the constructors within an
// ordinary struct.

// A constructor from the callsite's point of view is a function that returns the struct type.
// A constructor from the inside the body is treated as a function that modifies `this` and not return.

// In the following test-cases where the constructor either returns an expr or no value,
// such as `return ...` within a condition for early exit or not,
// It automatically gets resolved as
// `this = t;` along with
// `return this` if it returns the struct type, else `return`
RWStructuredBuffer<float> outputBuffer;

struct Impl
{
    float x;
    __init(float v)
    {
        if (v > 0)
        {
            this.x = 2 * v;
            return this;
        }
        else
        {
            this.x = 3 * v;
            return;
        }
    }

    __init(int ival)
    {
        float val = ival;
        Impl v = Impl(val);
        return v;
    }

    __init()
    {
        float val = 2.0;
        return Impl(val);
    }
}

[numthreads(1, 1, 1)]
void computeMain(uint id : SV_DispatchThreadID)
{
    // CHECK: 4.000000
    // CHECK: 2.000000
    // CHECK: -3.000000
    // CHECK: -6.000000
    outputBuffer[id] = Impl().x;
    outputBuffer[id + 1] = Impl(1).x;
    outputBuffer[id + 2] = Impl(-1).x;
    outputBuffer[id + 3] = Impl(-2.0).x;
}

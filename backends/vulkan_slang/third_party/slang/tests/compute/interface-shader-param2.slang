// interface-shader-param2.slang

// This test builds on `interface-shader-param.slang` by using
// concrete types that have data within them, instead of
// just empty types.

//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute

//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil
//DISABLED_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute

// A lot of the setup is the same as for `interface-shader-param`,
// so look there if you want the comments.

interface IRandomNumberGenerator
{
    [mutating]
    int randomInt();
}

interface IRandomNumberGenerationStrategy
{
    associatedtype Generator : IRandomNumberGenerator;
    Generator makeGenerator(int seed);
}

interface IModifier
{
    int modify(int val);
}

int test(
    int                             seed,
    IRandomNumberGenerationStrategy inStrategy,
    IModifier                       modifier)
{
    let strategy = inStrategy;
    var generator = strategy.makeGenerator(seed);
    let unused = generator.randomInt();
    let val = generator.randomInt();
    let modifiedVal = modifier.modify(val);
    return modifiedVal;
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out
RWStructuredBuffer<int> gOutputBuffer;

//TEST_INPUT:cbuffer(data=[0 0 0 0 1 0 0 0], stride=4):
ConstantBuffer<IRandomNumberGenerationStrategy> gStrategy;

[numthreads(4, 1, 1)]
void computeMain(

//TEST_INPUT:root_constants(data=[0 0 0 0 8 0 0 0], stride=4):
    uniform IModifier   modifier,
            uint3       dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;

    let inputVal : int = tid;
    let outputVal = test(inputVal, gStrategy, modifier);

    gOutputBuffer[tid] = outputVal;
}

// Okay, now we get to the part that is unique starting
// in this test: we add data to the concrete types
// that we will use as parameters.

struct MyStrategy : IRandomNumberGenerationStrategy
{
    int globalSeed;

    struct Generator : IRandomNumberGenerator
    {
        int state;

        [mutating]
        int randomInt()
        {
            return state++;
        }
    }

    Generator makeGenerator(int seed)
    {
        Generator generator = { seed ^ globalSeed };
        return generator;
    }
}

struct MyModifier : IModifier
{
    int localModifier;

    int modify(int val)
    {
        return val ^ localModifier;
    }
}

//TEST_INPUT: globalExistentialType MyStrategy
//TEST_INPUT: entryPointExistentialType MyModifier

// interface-shader-param-in-struct.slang

// This test puts interface-type shader parameters
// inside of structure types to make sure that works

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Passing, but: slang-test: Test context Slang session is leaking #5610
//DISABLE_TEST(compute):COMPARE_COMPUTE:-wgpu

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


//TEST_INPUT:set gOutputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<int> gOutputBuffer;

// Note: even though `C` doesn't include any
// uniform/ordinary dat as declared, it gets a
// constant buffer allocated for it because
// there is no way to rule out the possibility
// that it *will* contain uniform/ordinary data
// after specialization.
//
//TEST_INPUT:set C = new{ new MyStrategy{ ubuffer(data=[1 2 4 8], stride=4) } }
//TEST_INPUT: globalExistentialType MyStrategy
cbuffer C
{
   IRandomNumberGenerationStrategy gStrategy;
}

struct Stuff
{
    IModifier modifier;
    int extra;
}

[numthreads(4, 1, 1)]
void computeMain(
//TEST_INPUT:set stuff = { new MyModifier{ ubuffer(data=[16 32 64 128], stride=4) }, 256 }
//TEST_INPUT: entryPointExistentialType MyModifier
    uniform Stuff stuff,

    int3       dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;

    let inputVal : int = tid;
    let outputVal = test(inputVal, gStrategy, stuff.modifier)
        + stuff.extra*stuff.extra;

    gOutputBuffer[tid] = outputVal;
}

// Okay, now we get to the part that is unique starting
// in this test: we add data to the concrete types
// that we will use as parameters.

struct MyStrategy : IRandomNumberGenerationStrategy
{
    RWStructuredBuffer<int> globalSeeds;

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
        Generator generator = { globalSeeds[seed] };
        return generator;
    }
}

struct MyModifier : IModifier
{
    RWStructuredBuffer<int> localModifiers;

    int modify(int val)
    {
        return val ^ localModifiers[val & 3];
    }
}

// interface-shader-param3.slang

// This test builds on `interface-shader-param2.slang` by putting
// interface types at more complicated places in the overall layout.
//

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

ConstantBuffer<IRandomNumberGenerationStrategy> gStrategy;

// Note: The current strategy we use for laying out shader
// parameters in the presence of existential/interface types
// is to always put global-scope parameters before any
// entry-point parameters. As a result we need to provide
// the buffer for the specialized version of `gStrategy`
// here, and we will go ahead and provide the concrete
// type argument at the same time.
//
//TEST_INPUT: globalExistentialType MyStrategy
//TEST_INPUT:cbuffer(data=[0 0 0 0 1 0 0 0], stride=4):

[numthreads(4, 1, 1)]
void computeMain(

// We will be declaring two different `uniform` parameters in the
// entry-point parameter list, which will both get allocated to
// the same constant buffer.
//
// The first parameter will use an interface type, while the second
// will be plain-old-data.
//
    uniform IModifier   modifier,
    uniform int         extra,
//
// The computed layout for the entry-point constant buffer will
// always place the non-interface-type data first, so the first
// four bytes of our buffer represent the `extra` field.
//
// After all the non-interface-type data is laid out, we lay out
// the contents of extistential value slots in order, using the
// ordinary constant buffer packing rules. Because the concrete
// type we'll be plugging in for `modifier` is a `struct` type,
// it will need to start on a 16-byte-aligned boundary.
//
// Here's the incantation to make the test runner fill in the constant buffer:
//
//TEST_INPUT:root_constants(data=[0 0 0 0 16 0 0 0 256], stride=4):
//
// So, the value `256` will be used for `extra` and the value `16`
// will be written to the first four bytes of the concrete value
// being used for `modifier`.

            uint3       dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;

    let inputVal : int = tid;
    let outputVal = test(inputVal, gStrategy, modifier)
    	+ extra*extra;

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
        return val * localModifier;
    }
}

//TEST_INPUT: entryPointExistentialType MyModifier

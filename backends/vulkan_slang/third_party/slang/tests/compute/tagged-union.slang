// tagged-union.slang
//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12
//DISABLED_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute


// The goal of this test is to show that we can generate
// dynamic-dispatch logic from Slang code written using
// interface+generics, entirely from the compiler API side
// and without code changes.

// We are going to define a dummy interface just for testing
// purposes, that can be used to see which implementation
// got invoked at runtime.
//
interface IFrobnicator
{
    int frobnicate(int value);
}

// Now we will define two different implementations that
// "frobnicate" values differently. The first will just
// add to the value from a member variable.
//
struct A : IFrobnicator
{
    int a;
    int frobnicate(int value)
    {
        return value + a;
    }
}

// The second implementation will have an additional field
// of local state, and will do a tiny bit more math.
//
struct B : IFrobnicator
{
    int b;
    int y;
    int frobnicate(int value)
    {
        return value * b + y;
    }
}

[numthreads(4, 1, 1)]
void computeMain
// Now we will define the generic type parameter for our shader,
// which will be constraints to be a type that implements our
// `IFrobnicator` interface.
//
    <T : IFrobnicator>
//
// For the actual test runner, we will instruct it to plug in
// a tagged-union type over the two concrete implemetnations.
// Note that while we are writing this line in the source file,
// it is in comments so that the Slang compiler only knows about
// our intention when it is informed via the API.
//
//TEST_INPUT: type __TaggedUnion(A,B)
    (
// Next we need to pass in the actual parameter data for our
// chosen `IFrobnicator` implementation. The decalration of
// the constant buffer follows the conventional approach for
// using global generic parameters, so there is nothing much
// to say.
//
// Note: We are not using `ParameterBlock<_>` here because
// the `render-test` tool doesn't yet support code that
// uses multiple descriptor tables/sets.
//
    uniform ConstantBuffer<T> gFrobnicator,

// Where things get interesting is when we go to provide the
// data that will be used by the parameter block.
//
// Because we are plugging in a taggd union type, the actual
// data type that we fill in will be something kind of like:
//
//      struct S { union { A a; B b; }; uint tag; };
//
// When compiling for D3D, the size of that union will be
// 8 bytes (for the largest case), and so the tag will come
// as the third 32-bit value. We will initialize our data
// buffer for a `B` value then, which will get the tag `1`
// since it was the second option in our `__TaggedUnion`.
//
// In the Vulkan case the size of both `A` and `B` is 16 bytes
// (because they round up structure sizes to their alignment)
// and so the tag value will be the fifth 32-bit value.
// We will thus set up the same data buffer to look like an `A`
// value to Vulkan!
//
//TEST_INPUT: cbuffer(data=[16 9 1 0 0], stride=4):


//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out
    uniform RWStructuredBuffer<int> gOutputBuffer,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    int val = tid;
    val = gFrobnicator.frobnicate(val);

    gOutputBuffer[tid] = val;
}

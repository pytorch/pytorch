// interface-shader-param-legalization.slang

// Test case where concrete type implementing
// an interface has resource-type fields nested in it.

//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute

interface IModifier
{
    int modify(int val);
}

// Note: a constant buffer for the global scope gets introduced
// at this point, since `gModifier` introduces uniform storage
// for the RTTI, witness table, and "any value" parts of the
// existential.
//
//TEST_INPUT:cbuffer(data=[1 2 3 4 5 6 7 8], stride=4):
uniform IModifier gModifier;

int test(
    int val)
{
    return gModifier.modify(val);
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out
RWStructuredBuffer<int> gOutputBuffer;

[numthreads(4, 1, 1)]
void computeMain(
    uint3       dispatchThreadID : SV_DispatchThreadID)
{
    let tid = dispatchThreadID.x;

    let inputVal : int = tid;
    let outputVal = test(inputVal);

    gOutputBuffer[tid] = outputVal;
}

// Now that we've define all the logic of the entry point,
// we will define some concrete types that we can plug
// in for the interface-type parameters.

struct ConcreteData
{
    int offset;
}

struct ConcreteModifier : IModifier
{
    ConstantBuffer<ConcreteData> data;

    int modify(int val) { return val + data.offset; }
}

//TEST_INPUT: globalExistentialType ConcreteModifier
//TEST_INPUT:cbuffer(data=[256], stride=4):out

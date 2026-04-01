// extern-test.slang

//TEST:COMPILE: -no-codegen tests/serialization/extern/module-a.slang -o tests/serialization/extern/module-a.slang-lib
//TEST:COMPILE: -no-codegen tests/serialization/extern/module-b.slang -o tests/serialization/extern/module-b.slang-lib
//TEST:COMPARE_COMPUTE_EX: -slang -compute -xslang -r -xslang tests/serialization/extern/module-a.slang-lib -xslang -r -xslang tests/serialization/extern/module-b.slang-lib -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// Declare the type exists
extern struct Thing {};
// A mechanism to make a Thing without knowing the specific fields.
extern Thing makeThing(int a, int b);

extern int doSomething(Thing a, Thing b);

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = int(dispatchThreadID.x);

    Thing a = makeThing(index + 1, index + 2);
    Thing b = makeThing(-index , index * 2);
    
    outputBuffer[index] = doSomething(a, b);
}
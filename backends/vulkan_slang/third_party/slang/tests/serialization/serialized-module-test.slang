// serialized-module-test.slang

// A test to try out the basics of module
// serialization.

//TEST:COMPILE: tests/serialization/serialized-module.slang -o tests/serialization/serialized-module.slang-module
//TEST:COMPARE_COMPUTE_EX:-slang -compute -xslang -r -xslang tests/serialization/serialized-module.slang-module -shaderobj

//import serialized_module;

// This is fragile - needs match the definition in serialized_module
import serialized_module_shared;

extern int foo(Thing thing);

//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Thing thing;

    int index = (int)dispatchThreadID.x;
        
    thing.a = index;
    thing.b = -index;

    outputBuffer[index] = foo(thing);
}

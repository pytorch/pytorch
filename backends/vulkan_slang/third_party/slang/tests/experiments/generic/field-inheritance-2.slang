//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* The test is for behavior with inheritance and two fields with the same name. 

.slang(24): error 39999: no overload for 'int' applicable to arguments of type (overload group)
    outputBuffer[index] = 1 + int(a.a);

Looks like the problem is that it sees 'a' as 'overloaded' - which I guess is one way to look at it. It is then followed with ...
    
core.meta.slang(230): note 39999: candidate: int.init(uint64_t)
core.meta.slang(230): note 39999: candidate: int.init(uint)

Which doesn't seem appropriate    
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyStruct
{
    int a = 10;
};

struct MyStruct2 : MyStruct
{
    int a = 10;
};    

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    MyStruct2 a;

	outputBuffer[index] = 1 + int(a.a);
}


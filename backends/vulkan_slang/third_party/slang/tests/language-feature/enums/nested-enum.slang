// nested-num.slang

// Test enums defined nested in a struct work as expected. 

//TEST(compute):COMPARE_COMPUTE: -shaderobj

struct Outer
{
    enum Channel
    {
        Red,
        Green,
        Blue,
        Alpha,
    }
    
    static int doSomething(int v) { return v + 1; }
    typedef int SomeType;
    
    static int someValue = 10;
    
    static int getHeuristicResult()
    {
        // This tests that the cast heuristic works here. 
        // when the compiler cannot determine if it's a type or not at this point, 
        // because these declarations have not been seen.
            
        // Has whitespace (between + and the next thing) -> must be an expression
        int value = (Inner::anotherValue) + 1;
        // No whitespace, so assumed to be a cast
        Inner::Enum anotherValue = (Inner::Enum) +1;
        
        return Inner::doSomethingElse(value + (int)anotherValue);
    }
    
    struct Inner
    {
        enum Enum
        {
            A,
            B,
        };
        
        static int anotherValue = 10;
        static int doSomethingElse(int a) { return a - 1; }
    }
}

int test(int val)
{
    Outer::Channel channel = (Outer::Channel)val; // Outer::Channel(val);
    Outer::Channel otherChannel = channel; // (Outer::Channel)val;

    typedef Outer::Channel Channel;

    int result = 0;
    if(channel == Channel.Red)          result += 1;
    if(channel != Channel.Green)        result += 16;
    if(otherChannel == Channel.Blue)    result += 16*16;
    if(otherChannel != Channel.Alpha)   result += 16*16*16;

    return result;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int value = (Outer::someValue) + 1 + Outer::getHeuristicResult();
    Outer::Channel anotherValue = (Outer::Channel) +1;
    
    // These work because the type can be determined because they are already declared at this point
    // Check can see this is a function call
    value = (Outer::doSomething)(value);
    // Check can see this is a cast
    value += (Outer::SomeType)(value);
    
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal) + value * 2 + int(anotherValue) * 4;
    outputBuffer[tid] = outVal;
}

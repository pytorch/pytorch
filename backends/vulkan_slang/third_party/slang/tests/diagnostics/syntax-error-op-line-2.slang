// syntax-error-op-line-2.slang

// NOTE! That although this is a 'diagnostic' like test, it tests using downstream compiler
// the downstream compiler being present is a requirement, so we mark as a 'TEST' so that 
// those tests are made.

//TEST:SIMPLE_LINE:-entry computeMain -target spirv -emit-spirv-via-glsl
//TEST:SIMPLE_LINE:-entry computeMain -target dxil -profile cs_6_0 
//TEST:SIMPLE_LINE:-entry computeMain -target dxbc 
//TEST:SIMPLE_LINE:-entry computeMain -target shader-dll 
//TEST:SIMPLE_LINE:-entry computeMain -target ptx 

// Here the thing being checked is error reporting around return, and += 

int doSomething(int a)
{
    a += a;
    
    return a 
    += 
    __SyntaxError();
}

[shader("compute")]
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int id = int(dispatchThreadID.x);
    
    int v = int(dispatchThreadID.y);
    
    v += doSomething(id);
}
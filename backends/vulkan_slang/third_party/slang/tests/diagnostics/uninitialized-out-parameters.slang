//TEST:SIMPLE(filecheck=CHK): -target spirv -entry computeMain

// Using before assigning
float regular_undefined_use(out float3 v)
{
    //CHK-DAG: ([[# @LINE + 1]]): warning 41015: use of uninitialized out parameter 'v'
    float r = v.x + 1.0;
    
    //CHK-DAG: ([[# @LINE + 1]]): warning 41018: returning without initializing out parameter 'v'
    return r;
}

// Returning before assigning
float returning_undefined_use(out float x)
{
    //CHK-DAG: ([[# @LINE + 1]]): warning 41018: returning without initializing out parameter 'x'
    return 0;
}

// Implicit, still returning before assigning
void implicit_undefined_use(out float x) 
{
    //CHK-DAG: ([[# @LINE + 1]]): warning 41018: returning without initializing out parameter 'x'
}

// Warn on potential return paths
void control_flow_undefined(bool b, out float y)
{
    if(b)
    {
        //CHK-DAG: ([[# @LINE + 1]]): warning 41018: returning without initializing out parameter 'y'
        return;
    }
    y = 0;
    return;
}

// No warnings if all paths are fine
void control_flow_defined(bool b, out float y)
{
    if(b)
    {
        unused(y);
        return;
    }
    y = 0;
    return;
}

// Specialized handling for intrinsic asm
void instrincs_defined(float x, out float exp)
{
    __intrinsic_asm "frexp";
}

// Specialized handling for target switches
void target_switching_undefined(float x, out float exp)
{
    //CHK-DAG: ([[# @LINE + 2]]): warning 41018: returning without initializing out parameter 'exp'
    __target_switch {}
}

void target_switching_defined(float x, out float exp)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "frexp";
    }
}

void target_switching_blocks(uint x, out half v)
{
    __target_switch
    {
    case hlsl:
        if ((x & 2) == 0)
            v = half(1);
        else
            v = half(1);
        return;
    case glsl:
    case spirv:
        {
            if ((x & 2) == 0)
                v = half(1);
            else
                v = half(1);
            return;
        }
    }
}

//CHK-NOT: warning 41015
//CHK-NOT: warning 41018

[Shader("compute")]
[NumThreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
}


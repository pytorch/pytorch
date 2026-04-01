//TEST:SIMPLE(filecheck=CHECK): -target cpp -entry computeMain -stage compute -allow-glsl

// CHECK: computeMain(

layout(binding = 0) buffer MyBlockName
{
    int v[1];
} outputBuffer;

// `default` only should try to define `glsl`, but glsl target is 
// present in a `case` statment. Therefore this code should error
// since cpp is missing a target for a function called
[require(cpp)]
[require(glsl)]
[require(spirv)]
void someInternalFunc()
{
    __target_switch
    {
    case glsl:
        outputBuffer.v[0] = 0;
    default:
        outputBuffer.v[0] = 0;
    }
}
void someMin()
{
    someInternalFunc();
}
[numthreads(1,1,1)]
void computeMain()
{
    someMin();
}

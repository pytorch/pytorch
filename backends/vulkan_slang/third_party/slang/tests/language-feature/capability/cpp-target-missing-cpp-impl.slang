//TEST:SIMPLE(filecheck=CHECK): -target cpp -entry computeMain -stage compute -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target cpp -entry computeMain -stage compute -allow-glsl -ignore-capabilities
// CHECK_IGNORE_CAPS-NOT: error 36107
// CHECK: error 36107

layout(binding = 0) buffer MyBlockName
{
    int v[1];
} outputBuffer;

// `default` only should try to define `glsl`, but glsl target is 
// present in a `case` statment. Therefore this code should error
// since cpp is missing a target for a function called
[require(glsl)]
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

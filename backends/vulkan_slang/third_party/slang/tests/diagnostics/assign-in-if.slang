//DIAGNOSTIC_TEST:REFLECTION:-stage compute -entry main -target hlsl

[numthreads(1, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int a = 1;
    if (a = 0) // error
    {}
    if ((a = 0)) // ok.
    {}
}
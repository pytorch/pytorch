//DIAGNOSTIC_TEST:SIMPLE:

[numthreads(1, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int a = 1;
    a == 2; // warn
    (a == 2); // ok.
}
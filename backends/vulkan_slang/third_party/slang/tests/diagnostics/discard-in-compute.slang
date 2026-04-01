//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry main -profile cs_6_1
void test()
{
    discard; // This should lead to `test` having `fragment` capability requirement.
}

[shader("compute")]
[numthreads(1,1,1)]
void main()
{
    // CHECK: error 36107
    test(); // compute shader cannot call `test` that require capabiltiy `fragment`.
}

//TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage compute -entry main
RWStructuredBuffer<float> result;
// CHECK-NOT: result{{.*}}[int(0)] = 1
// CHECK: result{{.*}}[int(0)] = 2
// CHECK-NOT: result{{.*}}[int(0)] = 1
[numthreads(1,1,1)]
void main()
{
    int ic = -1;
    uint a = 2;
    if (ic < a)
    {
        result[0] = 1;
    }
    else
    {
        // Should pick this branch according to C spec.
        result[0] = 2;
    }
}
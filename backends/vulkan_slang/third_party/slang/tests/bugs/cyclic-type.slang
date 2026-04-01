
//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-directly

// CHECK: error 41001:

struct TA
{
    TB tb;
}

struct TB
{
    TA ta[2];
}

[numthreads(1,1,1)]
void main()
{
    TA ta;
    ta.tb.ta[0] = ta;
}
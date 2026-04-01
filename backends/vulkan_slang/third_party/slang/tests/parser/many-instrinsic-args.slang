//TEST:SIMPLE(filecheck=HLSL): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none

// HLSL: bar((int(0)), (int(1)), (int(2)), (int(3)), (int(4)), (int(5)), (int(6)), (int(7)), (int(8)), (int(9)), (int(10)), (int(11)))

__target_intrinsic(hlsl, "bar($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)")
int foo(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7, int x8, int x9, int x10, int x11);

[numthreads(1, 1, 1)]
void computeMain()
{
    foo(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
}

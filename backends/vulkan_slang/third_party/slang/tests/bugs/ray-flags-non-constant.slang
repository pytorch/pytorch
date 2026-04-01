// ray-flags-non-constant.slang

// Regression test for a compiler crash occuring when a generic with integer
// value parameters is specialized to a non-constant (and hence invalid) value.

//TEST:SIMPLE:-target dxil-assembly -entry main -stage compute

void main()
{
    RAY_FLAG rayFlags = RAY_FLAG_CULL_FRONT_FACING_TRIANGLES | RAY_FLAG_CULL_NON_OPAQUE;
    RayQuery<rayFlags> query;
}

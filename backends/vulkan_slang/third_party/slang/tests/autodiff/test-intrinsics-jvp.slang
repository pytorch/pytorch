//TEST_IGNORE_FILE:

__exported import test_intrinsics;

[ForwardDerivative(pow_jvp)]
float pow_(float x, float n);
float pow_jvp(float x, float n, float dx, float dn)
{
    return dx * n * pow(x, n-1) + dn * pow(x, n) * log(x);
}

[ForwardDerivative(max_jvp)]
float max_(float x, float y);
float max_jvp(float x, float y, float dx, float dy)
{
    return (x > y) ? dx : dy;
}


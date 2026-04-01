//DIAGNOSTIC_TEST:SIMPLE:

[BackwardDerivative(__d_f)]
float f(float x)
{
    return x * x;
}

void __d_f(float x, float.Differential dout)
{
}

float main(float x)
{
    DifferentialPair<float> dpx = diffPair(x, 1.f);
    bwd_diff(f)(dpx, 1.f);
}
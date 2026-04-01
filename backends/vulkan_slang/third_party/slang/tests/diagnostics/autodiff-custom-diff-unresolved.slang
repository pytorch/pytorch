//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// Simple check to see that the compiler returns an error message if the
// custom derivative definition points to an undefined function.
//

void __d_f(float x, float.Differential dout)
{
}

[BackwardDerivative(__d_g)]
float f(float x)
{
    // CHECK:      tests/diagnostics/autodiff-custom-diff-unresolved.slang([[@LINE-3]]): error 30015: undefined identifier '__d_g'.
    // CHECK-NEXT:     [BackwardDerivative(__d_g)]
    return x * x;
}

float main(float x)
{
    DifferentialPair<float> dpx = diffPair(x, 1.f);
    bwd_diff(f)(dpx, 1.f);
}
//DIAGNOSTIC_TEST:SIMPLE:

float nonDiff(float x)
{
    return x;
}

[Differentiable]
float f(float x)
{
    return x * x;
}

[Differentiable]
float h(float x)
{
    float val = 0;

    // call to non-differentiable method 
    // (should error if the data-flow propagation works properly.)
    // 
    float y = nonDiff(x);

    // call to a differentiable method, using the
    // result of a non-differentiable call.
    // 
    val = f(y);

    return val;
}

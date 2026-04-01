//DIAGNOSTIC_TEST:SIMPLE:

float nonDiff(float x)
{
    return x;
}

[ForwardDifferentiable]
float f(float x)
{
    float val = 0;
    if (x > 5)
        val = x + 1;
    else
        val = nonDiff(x * 2.0f);
    // Not all path propagates derivatives through.
    return val;
}

// error: function does not return a differentiable value.
[ForwardDifferentiable]
void g(float x)
{
    float val = 0;
    if (x > 5)
        val = x + 1;

    for (int i = 0; i < 5; i++) // Not ok, we can't infer the loop iterations because the body modifies induction var.
    {
        i = (int)x;
        no_diff debugBreak();
    }
    return;
}


[ForwardDifferentiable]
float h(float x)
{
    float val = 0;
    // no diagnostic by clarifying intention.
    val = no_diff(nonDiff(x * 2.0f));
    return val;
}

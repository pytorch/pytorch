//DIAGNOSTIC_TEST:SIMPLE:


float doSomething(int x)
{
    for (int i = 0; i < x; i--) // warn.
    {}
    for (int i = 0; i < 5; i-=-2) // ok.
    {}
    for (int j = 0; j < 3; j += 0) // warn.
    {}
    for (int i = 0; i < 5; i++) // ok
    {
        for (int j = 0; j < 3; i++) // warn.
        {}
    }
    for (int i = 0; i < 5; i++) // ok
    {
        for (int j = 0; i < 4; j++) // warn.
        {}
    }

    [MaxIters(6)] // warn
    for (int i = 0; i <= 6; i+=3)
    {
    }

    [MaxIters(6)] // warn
    for (int i = 5; i >= 0; i -= 3)
    {
    }
    [MaxIters(6)] // warn
    for (int i = 5; i > 0; i--)
    {
    }

    [MaxIters(5)] // ok
    for (int i = 0; i < 5; i++) // ok
    {
    }

    for (int i = 1; i < 0; i++) // warn
    {
    }
    for (int i = 1; i >= 2; i--) // warn
    {
    }
    for (int i = 1; i >= 1; i--) // ok
    {
    }
    for (int i = 1; i > 1; i--) // warn
    {
    }
    [MaxIters(5)] // ok, because the loop body modifies i so we can't infer the iterations.
    for (int i = 0; i < 5; i+=2)
    {
        i--;
    }
    return 0.0;
}

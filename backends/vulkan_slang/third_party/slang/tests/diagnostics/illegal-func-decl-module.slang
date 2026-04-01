//TEST_IGNORE_FILE:

export float libraryFunction(float a)
{
    a = a + a;
    return a;
}

export float libraryFunction1(inout float a)
{
    a = a * a;
    return a;
}

export float libraryFunction2(inout float a, in float b, out float c)
{
    a = a * a;
    a = a + b;
    return a;
}

export float libraryFunction3(in float a)
{
    a = a + a;
    return a;
}

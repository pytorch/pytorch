//TEST_IGNORE_FILE:

// obfuscated-loc-module.slang


export int billy(int v)
{
    return v + 1;
}

// This function is designed to fail during IR passes/emit. 
export int silly(int a)
{
    int t = 0;
    [ForceUnroll(10)]
    while ( a > 0)
    {
        t = t + t + a;
    }
    return t;
}

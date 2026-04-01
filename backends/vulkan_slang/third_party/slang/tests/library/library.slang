//TEST_IGNORE_FILE:

// library.slang

export int doThing(int b)
{
    return b + b + 1;
}

public export int foo(int a) 
{ 
    return a * a + 1 + doThing(a + 2);
}


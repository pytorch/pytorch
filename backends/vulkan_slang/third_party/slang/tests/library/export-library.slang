//TEST_IGNORE_FILE:

// export-library.slang

int doThing(int b)
{
    return b + b + 1;
}

export int foo(int a) 
{ 
    return a * a + 1 + doThing(a + 2);
}


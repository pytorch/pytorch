//TEST_IGNORE_FILE:

// module-a.slang

export struct Thing
{
    int a; 
    int b;
};

export Thing makeThing(int a, int b)
{
    Thing thing;
    thing.a = a;
    thing.b = b;
    return thing;
}

export int foo(Thing thing) 
{ 
    return thing.a + thing.b * 2; 
}


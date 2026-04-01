//TEST_IGNORE_FILE:

// module-b.slang

// This looks like a definition (and it is) but with [__extern] it's definition will be replaced at link time with a defintion
extern struct Thing {};
extern int foo(Thing thing);

export int doSomething(Thing a, Thing b)
{
    return foo(a) + foo(b);
}



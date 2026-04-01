//DISABLE_TEST:SIMPLE:
// Disabled because we don't actually perform this inference yet

func foo(f : functype (float) -> int) -> int
{
    return f(0);
}

int boo<T>(T)
{
    return 1;
}

int zoo()
{
    // We should infer that we want boo<float>
    return foo(boo);
}

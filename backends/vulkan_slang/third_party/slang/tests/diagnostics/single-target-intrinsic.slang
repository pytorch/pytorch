//DIAGNOSTIC_TEST(windows):SIMPLE:


T doThing<T>(T in)
{
    return in;
}

__target_intrinsic(hlsl, "doSomethingIntrinsically($0)")
T doThing<T>(T in);

__target_intrinsic(hlsl, "letsRedefineIt($0)")
T doThing<T>(T in);

void test()
{
    int a = 5;
    int b = doThing(a);
}
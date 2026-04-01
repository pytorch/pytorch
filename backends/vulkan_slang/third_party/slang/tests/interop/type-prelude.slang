//TEST:EXECUTABLE:

[RequirePrelude(cpp, "const int PRELUDE = 2; struct Foo{};")]
[RequirePrelude(glsl, "INVALID_PRELUDE")]
__target_intrinsic(cpp, "Foo")
struct Foo
{
    int test()
    {
        __intrinsic_asm "(PRELUDE)";
    }
}

export __extern_cpp int main()
{
    Foo f;
    printf("%d\n", f.test());
    return 0;
}
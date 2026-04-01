//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target spirv -emit-spirv-directly -entry main -stage compute -ignore-capabilities
// CHECK_IGNORE_CAPS-NOT: error 36104
module test;

[require(spvAtomicFloat16AddEXT)]
interface IFoo
{
    [require(spvRayQueryKHR)]
    void method1();

    void method2();
}

[require(spvGroupNonUniformArithmetic)]
void useNonUniformArithmetic()
{}

[require(spvRayQueryKHR)]
void useRayQueryKHR()
{}

[require(spvAtomicFloat16AddEXT)]
void useAtomicFloat16()
{}

// This should be OK, uses nothing past what is declared in the interface.
struct Impl1 : IFoo
{
    void method1()
    {
        useAtomicFloat16();
        useRayQueryKHR();
    }

    void method2()
    {
        useAtomicFloat16();
    }
}

struct Impl2 : IFoo
{
    // CHECK: error 36104: {{.*}}spvGroupNonUniformArithmetic
    void method1()
    {
        useRayQueryKHR(); // OK.
        useNonUniformArithmetic(); // error.
    }

    // CHECK: error 36104: {{.*}}spvGroupNonUniformArithmetic
    void method2()
    {
        useAtomicFloat16();
        useNonUniformArithmetic(); // error.
    }
}

void main()
{}

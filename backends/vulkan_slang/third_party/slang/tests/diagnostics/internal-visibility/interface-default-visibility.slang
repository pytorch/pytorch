//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

module m;

public interface IFoo
{
    void foo(); // Should have public visibility by default.
}

public struct F : IFoo
{
    // CHECK:{{.*}}(13): error 30602:
    void foo() {};
}

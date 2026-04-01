//TEST:SIMPLE(filecheck=CHECK): -target spirv

struct Set<each T>
{
    Tuple<expand each T> data;
    void f(expand each T v){}
    void h<each U>(U x){}
    void g(expand each T d)
    {
        //CHECK-NOT: ([[# @LINE+1]]): error
        f(expand each d); // OK

        //CHECK-NOT: ([[# @LINE+1]]): error
        h(54); // OK, specializing free-form parameter U.

        //CHECK: ([[# @LINE+1]]): error
        f(); // error, cannot call f without arguments.

        //CHECK: ([[# @LINE+1]]): error
        f(5); // error, cannot call f with different type pack.
    }
}

[numthreads(1,1,1)]
void computeMain()
{
    Set<float> v;
}
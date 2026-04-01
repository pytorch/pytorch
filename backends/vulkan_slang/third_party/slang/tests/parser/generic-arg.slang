// generic-arg.slang

//DIAGNOSTIC_TEST:SIMPLE:

// Test disambiguation of expression and generic app.

namespace NS
{
    struct MyType<let u : int, let v : int>
    {
        int arr[u /(v+1-1)];
    }
}

StructuredBuffer<NS.MyType<1, 3>> buffer;

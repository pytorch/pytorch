//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK:      123
// CHECK-NEXT: 4567
// CHECK-NEXT: 0
// CHECK-NEXT: 0

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

struct S {
    int foo : 8;
    uint bar : 24;
};

// Generates the equivalent of this:
/*
struct S {
    int _backing;

    property foo : int
    {
        // int foo : 8;
        get
        {
            let backingWidth = 32;
            let fooWidth = 8;
            let topOfFoo = 8;
            // Shift left and then right to sign-extend foo properly
            return (int(_backing) << (backingWidth-topOfFoo)) >> (backingWidth-fooWidth);
        }
        [mutating] set(int x)
        {
            let fooMask = 0x000000FF;
            let bottomOfFoo = 0;
            _backing = int((_backing & ~fooMask) | ((int(x) << bottomOfFoo) & fooMask));
        }
    }

    // int bar : 24;
    property bar : int
    {
        get
        {
            let backingWidth = 32;
            let barWidth = 24;
            let topOfBar = 32;
            // Shift left and then right to sign-extend bar properly
            return (uint(_backing) << (backingWidth-topOfBar)) >> (backingWidth-barWidth);
        }
        [mutating] set(int x)
        {
            let barMask = 0xFFFFFF00;
            let bottomOfBar = 8;
            _backing = int((_backing & ~barMask) | ((int(x) << bottomOfBar) & barMask));
        }
    }
};
*/

[numthreads(1, 1, 1)]
void computeMain()
{
    S s;
    s.foo = 123;
    s.bar = 4567;
    outputBuffer[0] = s.foo;
    outputBuffer[1] = s.bar;

    s.foo = 0;
    s.bar = 0;
    outputBuffer[2] = s.foo;
    outputBuffer[3] = s.bar;
}

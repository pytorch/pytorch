
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj

import modulea;

// Definition of the Foo struct
// struct Foo
// {
//     int a;
//     int b;
// }

// This test demonstrates that whether struct is a C-Style type only depends on the definition of the struct, not the extension.
// Even we have an explicit constructor defined in extension of Foo, it is still a C-Style type.

extension Foo
{
    __init(int a)
    {
        this.a = a + 5;
        this.b = 3;
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{

    // This will miss both synthesized constructor and explicit constructor in extension, but it will still fall back
    // to legacy initializer list because Foo is a C-Style type.
    Foo foo = {};
    //CHECK: BUFFER: 0
    //CHECK: BUFFER-NEXT: 0
    outputBuffer[0] = foo.a;
    outputBuffer[1] = foo.b;


    // When providing 1 parameter, it will lookup for the explicit constructor in extension.
    Foo foo1 = {1};
    //CHECK: BUFFER: 6
    //CHECK: BUFFER-NEXT: 3
    outputBuffer[2] = foo1.a;
    outputBuffer[3] = foo1.b;
}

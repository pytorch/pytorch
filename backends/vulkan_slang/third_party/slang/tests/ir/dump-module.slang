// This tests slangc's -dump-module command.
// Dumping ".slang" should mean just that, and it should not automatically load up
// a ".slang-module" when available, because the intent of the -dump-module command
// is to see the file you requested. If there's a bug in slang-module output, it's
// important that -dump-module looks at the specific file you requested.

//TEST:COMPILE: tests/ir/dump-module.slang -o tests/ir/dump-module.slang-module -target spirv -embed-downstream-ir

//TEST:SIMPLE(filecheck=CHECK1): -dump-module tests/ir/dump-module.slang-module
//TEST:SIMPLE(filecheck=CHECK2): -dump-module tests/ir/dump-module.slang

module "export-library-generics";

public cbuffer Constants {
    public float x;
    public float y;
}

interface MyInterface
{
    int myMethod(int a);
}

struct MyType : MyInterface
{
    int myMethod(int a)
    {
        return a * 3;
    }
}

int genericFunc<T: MyInterface>(T arg)
{
    return arg.myMethod(3);
}

public int normalFuncUsesGeneric(int a)
{
    MyType obj;
    return genericFunc(obj);
}

public int normalFunc(int a, float b)
{
    return a - floor(b);
}

// CHECK1:EmbeddedDownstreamIR(6 : Int,
// CHECK1:               OpCapability Linkage

// CHECK2-NOT:EmbeddedDownstreamIR(6 : Int,
// CHECK2-NOT:               OpCapability Linkage


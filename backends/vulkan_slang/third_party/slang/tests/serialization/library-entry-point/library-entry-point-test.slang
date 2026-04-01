// library-entry-point-test.slang

//DISABLED_TEST:COMPILE: -module-name module -no-codegen -profile cs_5_0 -entry computeMain tests/serialization/library-entry-point/library-entry-point.slang -o tests/serialization/library-entry-point/library-entry-point.slang-lib
//DISABLED_TEST:COMPARE_COMPUTE_EX: -no-default-entry-point -xslang -module-name -xslang module -slang -compute -xslang -r -xslang tests/serialization/library-entry-point/library-entry-point.slang-lib

//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):out,name outputBuffer

int doSomething(int a)
{
    return a * a + 1;
}


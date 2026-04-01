//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that a module imported from an __include'd file can be used from other files
// in the same module, and that entry points and global parameters defined in an
// included file can be correctly discovered by the reflection API.

module c;

__include b; // uses helper module.
__include a; // imports helper module.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
//CHECK: 1

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv

// missing entrypoint attribute
void vertMain()
{}

// CHECK: error 57004
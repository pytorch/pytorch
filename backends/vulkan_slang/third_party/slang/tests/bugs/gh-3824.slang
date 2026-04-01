//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-directly

// CHECK: OpEntryPoint

namespace example {
  struct Example {}
}

struct Struct {
  example::Example example;
}

void func(Struct Struct)
{
    // Test parameter name to be the same as type name.
}

[numthreads(1,1,1)]
void main()
{}
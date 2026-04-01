//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

namespace foo::bar::baz {}
namespace foo::bar {} // <- leaving this out "fixes" the issue

// CHECK: OpEntryPoint

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{}
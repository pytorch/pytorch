//TEST(64-bit):REFLECTION(filecheck=CHECK): -target cuda

// CHECK: "binding": {"kind": "uniform", "offset": 0, "size": 8},
uniform RaytracingAccelerationStructure accel;

[numthreads(1,1,1)]
void main()
{}
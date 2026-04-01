//TEST:SIMPLE(filecheck=CHECK): -target cuda

// CHECK: optixReportIntersection(3.0f, 2U, int(1), int(2), int(3))

[shader("intersection")]
void test()
{
    ReportHitOptix(3.0f, 2, 1, 2, 3);
}
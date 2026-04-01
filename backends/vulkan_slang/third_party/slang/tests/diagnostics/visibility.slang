//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// CHECK-DAG:{{.*}}(4): error 30603
private struct MyStruct
{}

struct IS
{
    // CHECK-DAG:{{.*}}(10): error 30601
    public void fp();
}

// CHECK-DAG:{{.*}}(14): error 30604
public IS f() {}

public struct PS
{
    // CHECK-DAG:{{.*}}(19): error 30604
    public IS ii;

    // CHECK-DAG:{{.*}}(22): error 30604
    public property IS ii2 { get { return {}; } };
}

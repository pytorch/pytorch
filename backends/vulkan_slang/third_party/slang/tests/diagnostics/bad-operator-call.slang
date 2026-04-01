// bad-operator-call.slang

//DIAGNOSTIC_TEST(windows):SIMPLE(filecheck=CHECK):

// Test that bad calls to operators produce reasonable diagnostic messages.

// Note: This test is currently Windows-only because our Linux builds
// seem to print references to the core module code with paths that don't
// match the Windows build (which generated our baseline).

struct S {}

void test()
{
	int a;
    S b;
    // CHECK:{{.*}}.slang(18): error {{.*}}:
    a += b;
    // CHECK:{{.*}}.slang(20): error {{.*}}: no overload for '+' applicable to arguments of type (int, S)
    a = a + b;
    // CHECK:{{.*}}.slang(22): error {{.*}}: no overload for '~' applicable to arguments of type (S)
	a = ~b;

	vector<int, 4> c;
    vector<float, 3> d;
    // CHECK:{{.*}}.slang(27): error {{.*}}: argument passed to parameter '0' must be l-value.
	a += c;

    c = a + c;
    // CHECK:{{.*}}.slang(31): error {{.*}}: no overload for '+=' applicable to arguments of type (vector<float,3>, vector<int,4>)
    d += c;
    // CHECK:{{.*}}.slang(33): error {{.*}}: no overload for '+' applicable to arguments of type (vector<int,4>, vector<float,3>)
	d = c + d;
}

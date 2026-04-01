//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// Here we test that diagnostics appear once per usage of each of the
// deprecated declarations.

[deprecated("please use foo(bool)")]
void foo(int) {}

void foo(bool) {}

[deprecated("non-overloaded bar")]
void bar() {}

const float newPi = 3.1;

[deprecated("more precise approximations are available elsewhere")]
const int pi = 3;

[deprecated("no")]
struct Billiards
{
    [deprecated("puns")]
    bool bs[22];
}

int main()
{
    pi;
    // CHECK:      tests/diagnostics/deprecation.slang([[#@LINE-1]]): warning 31200: pi has been deprecated: more precise approximations are available elsewhere
    // CHECK-NEXT:     pi;
    // CHECK-NEXT:     ^~

    foo(0);
    // CHECK:      tests/diagnostics/deprecation.slang([[#@LINE-1]]): warning 31200: foo has been deprecated: please use foo(bool)
    // CHECK-NEXT:     foo(0);
    // CHECK-NEXT:     ^~~

    foo(false); // doesn't warn
    // CHECK-NOT:      tests/diagnostics/deprecation.slang([[#@LINE-1]])

    Billiards b;
    // CHECK:      tests/diagnostics/deprecation.slang([[#@LINE-1]]): warning 31200: Billiards has been deprecated: no
    // CHECK-NEXT:     Billiards b;
    // CHECK-NEXT:     ^~~~~~~~~

    b.bs;
    // CHECK:      tests/diagnostics/deprecation.slang([[#@LINE-1]]): warning 31200: bs has been deprecated: puns
    // CHECK-NEXT:     b.bs;
    // CHECK-NEXT:       ^~

    bar();
    // CHECK:      tests/diagnostics/deprecation.slang([[#@LINE-1]]): warning 31200: bar has been deprecated: non-overloaded bar
    // CHECK-NEXT:     bar();
    // CHECK-NEXT:     ^~~

    // Check we don't have any extra warnings not caught by the above diagnostics
    // CHECK-NOT: warning

    return 0;
}

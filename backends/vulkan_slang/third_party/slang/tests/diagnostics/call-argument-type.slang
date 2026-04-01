//DIAGNOSTIC_TEST(smoke):SIMPLE:
// call function with wrong argument type

struct A {};
struct B {};

void f(A a) {}
void g(B b)
{
	f(b);
}

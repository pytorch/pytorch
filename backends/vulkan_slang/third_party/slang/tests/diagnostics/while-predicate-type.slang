//DIAGNOSTIC_TEST:SIMPLE:
// bad type for `while` predicate

struct S {};

void foo()
{
	S s;
	while(s) {break;}
}

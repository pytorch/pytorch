//TEST:SIMPLE:
// test that we can `typedef` a type

typedef float F32;

F32 foo()
{
	float x = 123.0;
	return x;
}

float bar()
{
	return foo();
}

// call-instance-from-static.slang

// Test that calling an instance method from a static method is an error.

//DIAGNOSTIC_TEST:SIMPLE:-target hlsl -entry main -stage compute

struct Test
{
	void instanceMethod()
	{}

	static void staticMethod()
	{
		instanceMethod();
	}
}

void main()
{
	Test.staticMethod();
}
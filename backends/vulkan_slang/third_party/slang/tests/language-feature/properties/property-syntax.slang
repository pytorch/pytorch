// property-syntax.slang

// Confirm that property syntax is parsed correctly.

//TEST:SIMPLE:

struct Data
{
    int _a;
    int _b;

    // Traditional syntax
	property int a
	{
		get { return _a; }
		set(int newValue) { _a = newValue; }
	}

	// "Modern" syntax
	property b : int
	{
		get { return _b; }
		set(newValue : int) { _a = newValue; }
	}
}

int test(Data d)
{
	return d.a + d.b;
}

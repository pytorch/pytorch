// properties.slang

// Diagnostic messages related to property/subscript accessor declarations.

//DIAGNOSTIC_TEST:SIMPLE:

struct Test
{
	property p : int
	{
		get(a) { return 0; }

		set(a, b) { }
	}

	property q : float
	{
		set(c : int) { }
	}
}

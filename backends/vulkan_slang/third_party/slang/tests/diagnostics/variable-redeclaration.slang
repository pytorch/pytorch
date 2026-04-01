// variable-redeclaration.slang

//DIAGNOSTIC_TEST:SIMPLE:


// This test confirms that the compiler produces
// suitable diagnostics when variables are redeclared
// in a given scope.

// Global variables, including shader parameters

static int gA;

static Texture2D gA;

// Local variables

int testLocalRedeclaration(int x)
{
	int y = x;
	int y = x;
}

int testLocalShadowing(int x)
{
	int y = x;
	{
		// Because this declaration is in an inner
		// scope it should shadow the existing `y`
		// rather than conflcit with it.
		//
		// TODO: It would be reasonable for the
		// compiler to warn on this sort of code.
		//
		int y = x;
	}
}

// Structure fields
// Note: more diagnostics will be reported here because of the constructor synthesis
struct S
{
	int f;
	float f;
}

// Function parameter list

int testParameterRedeclaration(
	int 	size,
	float 	size)
{
	return size;
}

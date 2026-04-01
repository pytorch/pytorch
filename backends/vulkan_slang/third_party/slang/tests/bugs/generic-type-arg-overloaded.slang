// generic-type-arg-overloaded.slang

//DIAGNOSTIC_TEST:SIMPLE:

// Regression test to confirm that type checker
// doesn't crash when an overloaded identifier
// is used as a generic type argument.

interface IThing { int getVal(); }

struct Stuff : IThing { int getVal() { return 1; } }

// Conflicting declaration:
struct Stuff {}

int util<T : IThing>() { return 1; }

struct G {}
int nonGeneric() { return 2; }

int test()
{
	// This call should note the ambiguity,
	// rather than crash.
	//
	return util<Stuff>()

	// Adding an extra call to also test the
	// case of trying to speicalize something
	// like a generic when it isn't one.
	//
		+ nonGeneric<G>();
}

//TEST:SIMPLE:
// test that we can parse all the expected kinds of declarations

// global-scope `using` is another test

/* Note(tfoley): disabled during syntax transition
// pipeline
pipeline P
{

}
*/

// empty declaration
;

// struct type
struct Pair
{
	int head;
	float tail;

// Note(tfoley): semicolon is expected/required
// here for compatibility with HLSL (C-style
// declarators).
//
// TODO: this could be removed if we treat
// HLSL as a "compatibility mode"
};

// function at global scope
float tail(Pair p) { return p.tail; }

/* Note(tfoley): disabled during syntax transition

// module
module M
{
	// component declarations

	// using declarations
	
}

// a module can "inherit" from a pipeline
module M2
	targets P
{
}

// shader
shader S
{
	// component declarations

	// using declarations

}

*/
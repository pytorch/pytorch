//DIAGNOSTIC_TEST:SIMPLE:

// Forgetting to put in the `;` after declaraing a variable/field
// with a semantic used to put the compiler into an infinite loop.

struct Test
{
	float4 pos : SV_POSITION
};

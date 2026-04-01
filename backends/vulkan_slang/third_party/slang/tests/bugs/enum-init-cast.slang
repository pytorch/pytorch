// enum-init-cast.slang
//TEST:SIMPLE:

// Regression test for a bug around declaration checking order
// for `enum` types, which manifests when an `enum` case
// has an initializer that requires an implicit cast.

enum class AAA
{
	// These cases use a `uint` literal for initialization,
	// but the `enum` type will use `int` for its tags.

    BBB 	= 0u,
    CCC  	= 1u,
};

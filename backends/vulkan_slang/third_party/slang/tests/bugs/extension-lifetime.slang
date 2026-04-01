// extension-lifetime.slang

// This test is a regresion test for a bug where `extension`
// declarations are incorrectly being cached on the declarations
// they extend, so that an extension of a core module type (like `float`)
// ends up attaching a declaration from one compile request to that
// type, and then later compile requests that use the core module type
// try to look up through that extension even though (1) that
// shouldn't make sense semantically, and (2) that extension will
// have been deallocated when its parent compile request was
// destroyed.
//
// This test relies on the fact that our test runner uses a single
// slang compilation session (which loads the core module code) across
// multiple compilation tests. We can thus make this file contain
// two identical tests, with the knowledge that the second one
// will lead to the bad/crashing behavior if the first one ran
// and did a Bad Thing.

//TEST:SIMPLE:
//TEST:SIMPLE:

interface IThing
{
	float getThing();
}

extension float : IThing
{
	float getThing() { return this; }
}

float getThing<T : IThing>(T value)
{
	return value.getThing();
}

float test( float value )
{
	return getThing(value);
}

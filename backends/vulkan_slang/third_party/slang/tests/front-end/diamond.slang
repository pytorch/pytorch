// diamond.slang

//TEST:SIMPLE:

// Test to confirm that Slang can handle a "diamond"
// multiple inheritance pattern between interfaces,
// without having lookup ambiguity issues at use
// sites.

interface A { float getA(); }

interface B : A {}

interface C : A {}

interface D : B, C {}

float doIt<T : D>(T value)
{
	return value.getA();
}

struct Thing : D
{
	float getA() { return 0.0f; }
}

float test()
{
	Thing thing;
	return doIt(thing);
}
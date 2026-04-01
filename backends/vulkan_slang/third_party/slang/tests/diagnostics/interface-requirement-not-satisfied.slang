// interface-requirement-not-satisfied.slang

//DIAGNOSTIC_TEST:SIMPLE:

interface IFoo
{
	void bar();
}

struct T : IFoo
{	
}

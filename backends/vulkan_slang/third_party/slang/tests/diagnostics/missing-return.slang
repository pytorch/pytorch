// missing-return.slang

//DIAGNOSTIC_TEST:SIMPLE:

// Non-`void` function that fails to return

int bad(int a, int b)
{
	int result = a + b;

	// forgot `return` here
}

int alsoBad(int a, int b)
{
	if(a > b)
	{
		return a + b;
	}

	// forgot `return` here
}

int okay(int a, int b)
{
	int tmp = a;
	for(;;)
	{
		if(a > b)
			return tmp;

		a = b;
		b = tmp;
		tmp = a + b;
	}

	// Lack of `return` here is not
	// a problem, because we can never
	// actually get here
}

int alsoOkay(int a, int b)
{
	int tmp = a;
	while(true)
	{
		if(a > b)
			return tmp;

		a = b;
		b = tmp;
		tmp = a + b;
	}

	// Lack of `return` here is not
	// a problem, because we can never
	// actually get here
}
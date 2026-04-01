//DIAGNOSTIC_TEST:SIMPLE:-target dxbc -profile ps_5_0 -entry main

// Early versions of the front-end had bugs when local
// variables were used before their definition, or
// were defined using a reference to themselves.
//
// This file tries to ensure that we emit proper error
// diagnostics in these cases.

void usedBeforeDeclared()
{
    // b is used before it is declared
	float c = d + 1.0;
	int d = 0;
}

float4 main()
{
	usedBeforeDeclared();
	return 0;
}
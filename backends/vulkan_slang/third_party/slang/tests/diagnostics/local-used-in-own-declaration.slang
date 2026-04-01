//DIAGNOSTIC_TEST:SIMPLE:-target dxbc -profile ps_5_0

// Early versions of the front-end had bugs when local
// variables were used before their definition, or
// were defined using a reference to themselves.
//
// This file tries to ensure that we emit proper error
// diagnostics in these cases.

void usedInOwnDeclaration()
{
	int e = e;
}

float4 main()
{
	usedInOwnDeclaration();
	return 0;
}
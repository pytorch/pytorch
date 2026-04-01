//TEST_IGNORE_FILE:

// This file exists to be imported.
//
// It contains a semantic error that should lead to compilation failure
// in the module doing the importing.

void broken()
{
	int a = b;
}

// extension-import.slang

// Confirm that visibility of extensions through `import` is working.

//TEST:SIMPLE:

import extension_import_helper;

float test<T : IThing>(T value)
{
	return value.getValue();
}

float id(float x)
{
	return test(x);
}
//TEST:SIMPLE:

// Confirm that we correctly issue a diagnostic when
// we `import` a module that has some errors in it.

import import_with_error_extra;

void main()
{
	broken();
}

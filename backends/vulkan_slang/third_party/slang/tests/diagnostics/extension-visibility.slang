// extension-visibility.slang

// Confirm that visibility of `extensions` is
// correctly scoped via `import`.

//DIAGNOSTIC_TEST:SIMPLE:

import extension_visibility_a;

// Note: not importing b:
// import extension_visibility_b;

import extension_visibility_c;

int shouldntWork(MyThing thing)
{
	return helper(thing);
}

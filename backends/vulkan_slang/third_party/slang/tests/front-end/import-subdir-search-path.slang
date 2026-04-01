//TEST:SIMPLE: -Itests/front-end/subdir

// Confirming import lookup via search paths work

__import import_subdir_a;

// Should realize it's the same thing
__import subdir.import_subdir_a;

float bar(float x) { return foo(x); }

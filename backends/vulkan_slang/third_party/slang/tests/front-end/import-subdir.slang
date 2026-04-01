//TEST:SIMPLE:

// Confirming that we can use "dot notation" to import via a subdirectory

__import subdir.import_subdir_a;

float bar(float x) { return foo(x); }
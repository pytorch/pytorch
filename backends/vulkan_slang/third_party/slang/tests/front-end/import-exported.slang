//TEST:SIMPLE:

// Confirming that we can use a re-exported function

// `a` imports `b` (which defines `foo`) and re-exports it
__import import_exported_a;

float bar(float x) { return foo(x); }
//DISABLE_TEST:SIMPLE: -Itests/front-end/subdir -dump-repro import-subdir-search-path-repro.slang-repro
//DISABLE_TEST:SIMPLE_EX: -extract-repro import-subdir-search-path-repro.slang-repro
//DISABLE_TEST:SIMPLE: -load-repro import-subdir-search-path-repro.slang-repro

// Confirming import lookup via search paths work

__import import_subdir_a;

// Should realize it's the same thing
__import subdir.import_subdir_a;

float bar(float x) { return foo(x); }
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// A file that recursively imports itself
// (including transitive cases) should be diagnosed.

// CHECK: ([[#@LINE+1]]): error 38200
import recursive_import_extra;

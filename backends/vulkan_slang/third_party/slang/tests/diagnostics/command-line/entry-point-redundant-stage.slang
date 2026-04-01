// entry-point-redundant-stage.slang

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-stage vertex -stage vertex

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-entry vsMain -stage vertex -stage vertex

// CHECK: was specified more than once

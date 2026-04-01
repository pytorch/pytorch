// x-arg-tool-unknown.slang

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-entry main -stage compute -o myKernel.dxbc -o myKernel.dxbc -Xthingy

// {{.*}}CHECK: downstream tool name not known, allowed names are

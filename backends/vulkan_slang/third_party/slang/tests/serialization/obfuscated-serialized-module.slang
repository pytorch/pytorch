//TEST_IGNORE_FILE:

// obfuscated-serialized-module.slang
import obfuscated_serialized_module_shared;

export int foo(Thing thing) 
{ 
    return (thing.a + thing.b) - thing.b; 
}


//TEST_IGNORE_FILE:

// serialized-module.slang

import serialized_module_shared;

export int foo(Thing thing) 
{ 
    return (thing.a + thing.b) - thing.b; 
}


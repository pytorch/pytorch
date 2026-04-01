//TEST:COMPILE: tests/bugs/serialization/gh-6892.slang -o tests/bugs/serialization/gh-6892.slang-module

// assertion failure when compiling to a slang module

void f()
{
    for (int i = 0; i < 1; ++i)
        ;
}

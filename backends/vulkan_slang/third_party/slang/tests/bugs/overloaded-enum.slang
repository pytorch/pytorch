//TEST:SIMPLE(filecheck=CHECK):

enum E
{
    A,
    A
    // CHECK: overloaded-enum.slang([[#@LINE-1]]): error 30200: declaration of 'A' conflicts with existing declaration
}

E e = E::A;

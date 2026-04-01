// TEST(smoke):CPP_COMPILER_EXECUTE:

#include <stdio.h>
#include <stdlib.h>

extern int thing;

int main(int argc, char** argv)
{
    printf("Hello World %d!\n", thing);
    return 0;
}

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    void *handle;
    char *error;

    handle = dlopen("libm.so", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

//TEST_IGNORE_FILE:

// module-library-pointer-param.slang

module "module-library-pointer-param";

public int ptrFunc(int* a)
{
    return *a;
}

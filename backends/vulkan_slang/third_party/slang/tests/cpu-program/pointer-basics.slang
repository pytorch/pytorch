//TEST:EXECUTABLE:
export __extern_cpp int main()
{
    uint2 value;
    int *pValue = (int*)&value;
    *pValue = 1;
    (*pValue)++;
    ++pValue[0];
    ++pValue;
    *pValue = 1;
    pValue = (int *)&value;
    int64_t ptrVal = int64_t(pValue);
    pValue = (int *)ptrVal;
    if (pValue
        && pValue != nullptr
        && ptrVal != 0
        && value[0] == 3
        && pValue[1] == 1)
        printf("Success\n");
    else
        printf("Fail\n");
    return 0;
}

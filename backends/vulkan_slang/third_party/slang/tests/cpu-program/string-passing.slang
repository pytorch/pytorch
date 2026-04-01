//TEST:EXECUTABLE:

void printer(NativeString string)
{
    printf(string);
}

void indirectPrinter(String string)
{
    printer(string);
}

export __extern_cpp int main()
{
    indirectPrinter("success\n");
    return 0;
}

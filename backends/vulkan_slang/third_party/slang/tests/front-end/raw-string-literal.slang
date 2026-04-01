// Test raw string literals.

//TEST:EXECUTABLE:
__target_intrinsic(cpp, R"delim(printf("%c", $0))delim")
void writeChar(int8_t ch);

void printLines(NativeString text)
{
    NativeString str = text;
    int8_t *ptr = bit_cast<Ptr<int8_t>>(str);
    for (int i = 0; i < text.length; i++)
    {
        if (ptr[i] != 13)
        {
            writeChar(ptr[i]);
        }
    }
}

export __extern_cpp int main()
{
    printLines(
        R"(This is line 1.
        This is line 2.
        "Hello World"
        )");
    return 0;
}

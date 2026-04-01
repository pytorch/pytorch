// Disbaled because Slang should perform its own semantic checking now
//TEST_IGNORE_FILE
//TEST:COMPARE_HLSL: -profile cs_5_0

#ifdef __SLANG__
__import import_overload_error;
#else

void foo(int a) {}
void foo(float b) {}

#endif

void main()
{
// Note(tfoley): futzing around with tokens to
// make sure error message gets reported at a
// consistent location between languages.
int a;
foo();
}

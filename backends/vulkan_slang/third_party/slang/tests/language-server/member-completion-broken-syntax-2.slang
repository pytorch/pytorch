//TEST:LANG_SERVER:
//COMPLETE:13,7
UNDEFINED_MACRO(myGlobalParameterDefinedWithMacro)
UNDEFINED_MACRO(myGlobalParameterDefinedWithMacro2)

struct MyType
{
    int getSum() { return 0; }
}

void m(MyType t)
{
    t.
}

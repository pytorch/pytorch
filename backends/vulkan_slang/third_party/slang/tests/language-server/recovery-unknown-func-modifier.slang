// Test that the parser can recover after unknown function modifier.
//TEST:LANG_SERVER:
UNKNOWN_MODIFIER UNKNOWN_TYPE apparentFunc(int x, int y)
{
    return 0;
}

//HOVER:9,10
void nextFunc()
{
    syntax_error();
}
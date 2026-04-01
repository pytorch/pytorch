//DIAGNOSTIC_TEST:SIMPLE:

int doSomething(int a)
{
    // No warning, literal will be interpreted as 64 bit.
    uint64_t c0 = 0x800000000;
    
    // No warning as top bits are just ignored
    int c1 = -1ll;
    
    int c2 = int(-1u);
    
    // Should sign extend 
    int c3 = 0x80000000;
    
    // No warning, hex literal will be interpreted as an unsigned 64 integer then signed with negative operator.
    int64_t c4 = -0xfffffffff;
    
    a += (int)c0 + c1 + c2;
    
    int64_t b = 0;

    // Ok
    b += 0x800000000ll;
    
    uint64_t c5 = -2ull;

    // Warning, integer literal is too large for signed 64 bit, must be interpreted as unsigned.
    uint64_t d0 = 18446744073709551615;

    // Warning, integer literal is too small for signed 64 bit, must be interpreted as unsigned.
    uint64_t d1 = -9223372036854775809;

    // This is INT64_MIN and valid negative signed integer, but warning will be emitted as negative(-) is scanned
    // separately in the lexer, and the positive literal portion will emit a warning.
    // The final value will still be correctly set as INT64_MIN.
    //
    // To not have this warning the lexer must scan the negative operator and number together.
    uint64_t d2 = -9223372036854775808;
    
    return a + int(b);
}


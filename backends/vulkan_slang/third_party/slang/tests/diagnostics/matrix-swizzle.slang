//DIAGNOSTIC_TEST:SIMPLE:

int doSomething(int a)
{
    int2x3 m1 = int2x3(0, 1, 2, 3, 4, 5);
    int3x2 m2 = int3x2(0, 1, 2, 3, 4, 5);

    int c = m1._14; // Out of bounds
    c = m1._32; 
    c = m2._m22;
    c = m2._;       // unfinished
    c = m2._m; 
    c = m2._1; 
    c = m2._m1;
    c = m2._m12_;
    int2 c2 = m1._m11_11;   // Mixing of 1 and 0-indexing
    c = m1._11_11_11_11_11; // More than 4 elements
    c = m1.x;       // Invalid character
    c = m1._x;
    c = m1.x123;

    return m1._11;
}


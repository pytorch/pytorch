//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

struct ExplicitCtor
{
   int x;
   int y;
   __init(int x)
   {
      this.x = x;
      this.y = 0;
   }
  // compiler does not synthesize any ctors.
}

void test()
{
   // CHECK: error 39999: too many arguments to call (got 2, expected 1)
   ExplicitCtor e = {1, 2}; // error, no ctor matches initializer list.

   // CHECK: error 39999: not enough arguments to call (got 0, expected 1)
   ExplicitCtor e1 = {};
}

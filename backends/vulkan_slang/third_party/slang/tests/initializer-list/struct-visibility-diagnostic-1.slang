//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

public struct Visibility
{
  internal int x;
  public int y = 0;
  // the compiler does not synthesize any ctor.
  // the compiler will try to synthesize:
  //     public __init(int y);
  // but then it will find that `x` cannot be initialized.
  // so this synthesis will fail and no ctor will be added
  // to the type.
}

void test()
{
    // CHECK: error 39999: too many arguments to call (got 2, expected 1)
    Visibility t1 = {1, 2}; // error, no matching ctor
}



//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

public struct Visibility
{
  internal int x;
  public int y = 5;
  // the compiler does not synthesize any ctor.
  // the compiler will try to synthesize:
  //     public __init(int y);
  // but then it will find that `x` cannot be initialized.
  // so this synthesis will fail and no ctor will be added
  // to the type.
}

void test()
{
    // CHECK: warning 41021: default initializer for 'Visibility' will not initialize field 'x'
    Visibility t1 = {};
}



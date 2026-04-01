//DISABLE_DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

struct PartialInit
{
  int x = 1;
  int y;
  // compiler synthesizes:
  // __init(int x, int y);
}

void test()
{
   // TODO: Because we have a legacy logic that will always convert the one arugment ctor call to
   // initializer list, and that initializer list will fall back to the legacy C-Style initialization.
   // We need to remove that logic.

   // CHECK: error 33070: expected a function, got 'typeof(PartialInit)'
  PartialInit p = PartialInit(2); // error, no ctor match.
}

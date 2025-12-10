extern void VerifyC(void);
#ifdef VERIFY_CXX
extern void VerifyCXX(void);
#endif
#include "VerifyFortran.h"
extern void VerifyFortran(void);

int main(void)
{
  VerifyC();
#ifdef VERIFY_CXX
  VerifyCXX();
#endif
  VerifyFortran();
  return 0;
}

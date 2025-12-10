#ifdef CHECK_FUNCTION_EXISTS

#  ifdef __cplusplus
extern "C"
#  endif
  char
  CHECK_FUNCTION_EXISTS(void);
#  ifdef __CLASSIC_C__
int main()
{
  int ac;
  char* av[];
#  else
int main(int ac, char* av[])
{
#  endif
  CHECK_FUNCTION_EXISTS();
  if (ac > 1000) {
    return *av[0];
  }
  return 0;
}

#else /* CHECK_FUNCTION_EXISTS */

#  error "CHECK_FUNCTION_EXISTS has to specify the function"

#endif /* CHECK_FUNCTION_EXISTS */

#ifdef CHECK_VARIABLE_EXISTS

extern int CHECK_VARIABLE_EXISTS;

#  ifdef __CLASSIC_C__
int main()
{
  int ac;
  char* av[];
#  else
int main(int ac, char* av[])
{
#  endif
  if (ac > 1000) {
    return *av[0];
  }
  return CHECK_VARIABLE_EXISTS;
}

#else /* CHECK_VARIABLE_EXISTS */

#  error "CHECK_VARIABLE_EXISTS has to specify the variable"

#endif /* CHECK_VARIABLE_EXISTS */

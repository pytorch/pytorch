static inline int static_foo()
{
  return 0;
}

int main(int argc, char *argv[])
{
  static_foo();
  return 0;
}

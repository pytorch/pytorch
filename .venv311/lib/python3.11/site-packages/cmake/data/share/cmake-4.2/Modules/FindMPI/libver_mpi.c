#include <mpi.h>

#ifdef __cplusplus
#  include <cstdio>
#else
#  include <stdio.h>
#endif

int main(int argc, char* argv[])
{
  char mpilibver_str[MPI_MAX_LIBRARY_VERSION_STRING];
  int mpilibver_len;
  MPI_Get_library_version(mpilibver_str, &mpilibver_len);
#ifdef __cplusplus
  std::puts(mpilibver_str);
#else
  puts(mpilibver_str);
#endif
  return 0;
}

#include <mpi.h>

#ifdef __cplusplus
#  include <cstdio>
#else
#  include <stdio.h>
#endif

#if defined(MPI_VERSION) && defined(MPI_SUBVERSION)
static char const mpiver_str[] = { 'I', 'N',
                                   'F', 'O',
                                   ':', 'M',
                                   'P', 'I',
                                   '-', 'V',
                                   'E', 'R',
                                   '[', ('0' + MPI_VERSION),
                                   '.', ('0' + MPI_SUBVERSION),
                                   ']', '\0' };
#endif

int main(int argc, char* argv[])
{
#if defined(MPI_VERSION) && defined(MPI_SUBVERSION)
#  ifdef __cplusplus
  std::puts(mpiver_str);
#  else
  puts(mpiver_str);
#  endif
#endif
#ifdef TEST_MPI_MPICXX
  MPI::MPI_Init(&argc, &argv);
  MPI::MPI_Finalize();
#else
  MPI_Init(&argc, &argv);
  MPI_Finalize();
#endif
  return 0;
}

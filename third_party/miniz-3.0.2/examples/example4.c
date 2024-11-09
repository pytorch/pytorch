// example4.c - Uses tinfl.c to decompress a zlib stream in memory to an output file
// Public domain, May 15 2011, Rich Geldreich, richgel99@gmail.com. See "unlicense" statement at the end of tinfl.c.
#include "miniz.h"
#include <stdio.h>
#include <limits.h>

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;

#define my_max(a,b) (((a) > (b)) ? (a) : (b))
#define my_min(a,b) (((a) < (b)) ? (a) : (b))

static int tinfl_put_buf_func(const void* pBuf, int len, void *pUser)
{
  return len == (int)fwrite(pBuf, 1, len, (FILE*)pUser);
}

int main(int argc, char *argv[])
{
  int status;
  FILE *pInfile, *pOutfile;
  uint infile_size, outfile_size;
  size_t in_buf_size;
  uint8 *pCmp_data;
  long file_loc;

  if (argc != 3)
  {
    printf("Usage: example4 infile outfile\n");
    printf("Decompresses zlib stream in file infile to file outfile.\n");
    printf("Input file must be able to fit entirely in memory.\n");
    printf("example3 can be used to create compressed zlib streams.\n");
    return EXIT_FAILURE;
  }

  // Open input file.
  pInfile = fopen(argv[1], "rb");
  if (!pInfile)
  {
    printf("Failed opening input file!\n");
    return EXIT_FAILURE;
  }

  // Determine input file's size.
  fseek(pInfile, 0, SEEK_END);
  file_loc = ftell(pInfile);
  fseek(pInfile, 0, SEEK_SET);

  if ((file_loc < 0) || ((mz_uint64)file_loc > INT_MAX))
  {
     // This is not a limitation of miniz or tinfl, but this example.
     printf("File is too large to be processed by this example.\n");
     return EXIT_FAILURE;
  }

  infile_size = (uint)file_loc;

  pCmp_data = (uint8 *)malloc(infile_size);
  if (!pCmp_data)
  {
    printf("Out of memory!\n");
    return EXIT_FAILURE;
  }
  if (fread(pCmp_data, 1, infile_size, pInfile) != infile_size)
  {
    printf("Failed reading input file!\n");
    return EXIT_FAILURE;
  }

  // Open output file.
  pOutfile = fopen(argv[2], "wb");
  if (!pOutfile)
  {
    printf("Failed opening output file!\n");
    return EXIT_FAILURE;
  }

  printf("Input file size: %u\n", infile_size);

  in_buf_size = infile_size;
  status = tinfl_decompress_mem_to_callback(pCmp_data, &in_buf_size, tinfl_put_buf_func, pOutfile, TINFL_FLAG_PARSE_ZLIB_HEADER);
  if (!status)
  {
    printf("tinfl_decompress_mem_to_callback() failed with status %i!\n", status);
    return EXIT_FAILURE;
  }

  outfile_size = ftell(pOutfile);

  fclose(pInfile);
  if (EOF == fclose(pOutfile))
  {
    printf("Failed writing to output file!\n");
    return EXIT_FAILURE;
  }

  printf("Total input bytes: %u\n", (uint)in_buf_size);
  printf("Total output bytes: %u\n", outfile_size);
  printf("Success.\n");
  return EXIT_SUCCESS;
}

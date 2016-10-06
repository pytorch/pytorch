#include <TH.h>
#include "THIMG.h"


/* Get the next character in the file, skipping over comments, which
 * start with a # and continue to the end of the line.
 */
static char ppm_getc(FILE *fp)
{
   char ch;

   ch = (char)getc(fp);
   if (ch == '#') {
      do {
         ch = (char)getc(fp);
      } while (ch != '\n' && ch != '\r');
   }

   return ch;
}


/* Get the next integer, skipping whitespace and comments. */
static long ppm_get_long(FILE *fp)
{
   char ch;
   long i = 0;

   do {
      ch = ppm_getc(fp);
   } while (ch == ' ' || ch == ',' || ch == '\t' || ch == '\n' || ch == '\r');

   do {
      i = i * 10 + ch - '0';
      ch = ppm_getc(fp);
   } while (ch >= '0' && ch <= '9');

   return i;
}


#include "generic/ppm.c"
#include "THGenerateAllTypes.h"

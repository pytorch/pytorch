// example6.c - Demonstrates how to miniz's PNG writer func
// Public domain, April 11 2012, Rich Geldreich, richgel99@gmail.com. See "unlicense" statement at the end of tinfl.c.
// Mandlebrot set code from http://rosettacode.org/wiki/Mandelbrot_set#C
// Must link this example against libm on Linux.

// Purposely disable a whole bunch of stuff this low-level example doesn't use.
#define MINIZ_NO_STDIO
#define MINIZ_NO_TIME
#define MINIZ_NO_ZLIB_APIS
#include "miniz.h"

// Now include stdio.h because this test uses fopen(), etc. (but we still don't want miniz.c's stdio stuff, for testing).
#include <stdio.h>
#include <limits.h>
#include <math.h>

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;

typedef struct
{
  uint8 r, g, b;
} rgb_t;

static void hsv_to_rgb(int hue, int min, int max, rgb_t *p)
{
  const int invert = 0;
  const int saturation = 1;
  const int color_rotate = 0;

  if (min == max) max = min + 1;
  if (invert) hue = max - (hue - min);
  if (!saturation) {
    p->r = p->g = p->b = 255 * (max - hue) / (max - min);
    return;
  }
  double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
  double c = 255.0f * saturation;
  double X = c * (1 - fabs(fmod(h, 2) - 1));

  p->r = p->g = p->b = 0;

  switch((int)h) {
  case 0: p->r = c; p->g = X; return;
  case 1:	p->r = X; p->g = c; return;
  case 2: p->g = c; p->b = X; return;
  case 3: p->g = X; p->b = c; return;
  case 4: p->r = X; p->b = c; return;
  default:p->r = c; p->b = X;
  }
}

int main(int argc, char *argv[])
{
  (void)argc, (void)argv;

  // Image resolution
  const int iXmax = 4096;
  const int iYmax = 4096;

  // Output filename
  static const char *pFilename = "mandelbrot.png";

  int iX, iY;
  const double CxMin = -2.5;
  const double CxMax = 1.5;
  const double CyMin = -2.0;
  const double CyMax = 2.0;

  double PixelWidth = (CxMax - CxMin) / iXmax;
  double PixelHeight = (CyMax - CyMin) / iYmax;

  // Z=Zx+Zy*i  ;   Z0 = 0
  double Zx, Zy;
  double Zx2, Zy2; // Zx2=Zx*Zx;  Zy2=Zy*Zy

  int Iteration;
  const int IterationMax = 200;

  // bail-out value , radius of circle
  const double EscapeRadius = 2;
  double ER2=EscapeRadius * EscapeRadius;

  uint8 *pImage = (uint8 *)malloc(iXmax * 3 * iYmax);

  // world ( double) coordinate = parameter plane
  double Cx,Cy;

  int MinIter = 9999, MaxIter = 0;

  for(iY = 0; iY < iYmax; iY++)
  {
    Cy = CyMin + iY * PixelHeight;
    if (fabs(Cy) < PixelHeight/2)
      Cy = 0.0; // Main antenna

    for(iX = 0; iX < iXmax; iX++)
    {
      uint8 *color = pImage + (iX * 3) + (iY * iXmax * 3);

      Cx = CxMin + iX * PixelWidth;

      // initial value of orbit = critical point Z= 0
      Zx = 0.0;
      Zy = 0.0;
      Zx2 = Zx * Zx;
      Zy2 = Zy * Zy;

      for (Iteration=0;Iteration<IterationMax && ((Zx2+Zy2)<ER2);Iteration++)
      {
        Zy = 2 * Zx * Zy + Cy;
        Zx =Zx2 - Zy2 + Cx;
        Zx2 = Zx * Zx;
        Zy2 = Zy * Zy;
      };

      color[0] = (uint8)Iteration;
      color[1] = (uint8)Iteration >> 8;
      color[2] = 0;

      if (Iteration < MinIter)
        MinIter = Iteration;
      if (Iteration > MaxIter)
        MaxIter = Iteration;
    }
  }

  for(iY = 0; iY < iYmax; iY++)
  {
    for(iX = 0; iX < iXmax; iX++)
    {
      uint8 *color = (uint8 *)(pImage + (iX * 3) + (iY * iXmax * 3));

      uint Iterations = color[0] | (color[1] << 8U);

      hsv_to_rgb(Iterations, MinIter, MaxIter, (rgb_t *)color);
    }
  }

  // Now write the PNG image.
  {
    size_t png_data_size = 0;
    void *pPNG_data = tdefl_write_image_to_png_file_in_memory_ex(pImage, iXmax, iYmax, 3, &png_data_size, 6, MZ_FALSE);
    if (!pPNG_data)
      fprintf(stderr, "tdefl_write_image_to_png_file_in_memory_ex() failed!\n");
    else
    {
      FILE *pFile = fopen(pFilename, "wb");
      fwrite(pPNG_data, 1, png_data_size, pFile);
      fclose(pFile);
      printf("Wrote %s\n", pFilename);
    }

    // mz_free() is by default just an alias to free() internally, but if you've overridden miniz's allocation funcs you'll probably need to call mz_free().
    mz_free(pPNG_data);
  }

  free(pImage);

  return EXIT_SUCCESS;
}

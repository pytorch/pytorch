#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/unfold.c"
#else

#ifdef _WIN32
# include <windows.h>
#endif

/* note: due to write issues, this one cannot be parallelized as well as unfolded_copy */
void THNN_(unfolded_acc)(
          THTensor *finput,
          THTensor *input,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int nInputPlane,
          int inputWidth,
          int inputHeight,
          int outputWidth,
          int outputHeight)
{
#ifdef _WIN32
  LONG_PTR nip;
#else
  size_t nip;
#endif

  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(nip)
  for(nip = 0; nip < nInputPlane; nip++)
  {
    size_t kw, kh, y, x;
    long long ix = 0, iy = 0;
    for(kh = 0; kh < kH; kh++)
    {
      for(kw = 0; kw < kW; kw++)
      {
        real *src = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
        real *dst = input_data + nip*(inputHeight*inputWidth);
        if (padW > 0 || padH > 0) {
          size_t lpad,rpad;
          for(y = 0; y < outputHeight; y++) {
            iy = (long long)(y*dH - padH + kh);
            if (iy < 0 || iy >= inputHeight) {
            } else {
              if (dW==1){
                 ix = (long long)(0 - padW + kw);
                 lpad = fmaxf(0,(int)(padW-kw));
                 rpad = fmaxf(0,(int)(padW-(kW-kw-1)));
                 THVector_(add)(dst+(size_t)(iy*inputWidth+ix+lpad), src+(size_t)(y*outputWidth+lpad), 1, outputWidth - lpad - rpad); /* note: THVector_add could handle 1 value better */
              }
              else{
                for (x=0; x<outputWidth; x++){
                   ix = (long long)(x*dW - padW + kw);
                   if (ix < 0 || ix >= inputWidth){
                   }else
                     THVector_(add)(dst+(size_t)(iy*inputWidth+ix), src+(size_t)(y*outputWidth+x), 1, 1);
                }
              }
            }
          }
        } else {
          for(y = 0; y < outputHeight; y++) {
            iy = (long long)(y*dH + kh);
            ix = (long long)(0 + kw);
            if (dW == 1 )
               THVector_(add)(dst+(size_t)(iy*inputWidth+ix), src+(size_t)(y*outputWidth), 1, outputWidth); /* note: THVector_add could handle 1 value better */
            else{
              for(x = 0; x < outputWidth; x++)
                THVector_(add)(dst+(size_t)(iy*inputWidth+ix+x*dW), src+(size_t)(y*outputWidth+x), 1, 1);
            }
          }
        }
      }
    }
  }
}

void THNN_(unfolded_copy)(
          THTensor *finput,
          THTensor *input,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int nInputPlane,
          int inputWidth,
          int inputHeight,
          int outputWidth,
          int outputHeight)
{
  long k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane*kH*kW; k++) {
    size_t nip = k / (kH*kW);
    size_t rest = k % (kH*kW);
    size_t kh = rest / kW;
    size_t kw = rest % kW;
    size_t x,y;
    long long ix,iy;
    real *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
    real *src = input_data + nip*(inputHeight*inputWidth);
    if (padW > 0 || padH > 0) {
      size_t lpad,rpad;
      for(y = 0; y < outputHeight; y++) {
        iy = (long long)(y*dH - padH + kh);
        if (iy < 0 || iy >= inputHeight) {
          memset(dst+y*outputWidth, 0, sizeof(real)*outputWidth);
        } else {
          if (dW==1){
             ix = (long long)(0 - padW + kw);
             lpad = fmaxf(0,(int)(padW-kw));
             rpad = fmaxf(0,(int)(padW-(kW-kw-1)));
             if (outputWidth-rpad-lpad <= 0) {
                memset(dst+(size_t)(y*outputWidth), 0, sizeof(real)*outputWidth);
             } else {
                if (lpad > 0) memset(dst+y*outputWidth, 0, sizeof(real)*lpad);
                memcpy(dst+(size_t)(y*outputWidth+lpad), src+(size_t)(iy*inputWidth+ix+lpad), sizeof(real)*(outputWidth-rpad-lpad));
                if (rpad > 0) memset(dst+y*outputWidth + outputWidth - rpad, 0, sizeof(real)*rpad);
             }
          }
          else{
            for (x=0; x<outputWidth; x++){
               ix = (long long)(x*dW - padW + kw);
               if (ix < 0 || ix >= inputWidth)
                 memset(dst+(size_t)(y*outputWidth+x), 0, sizeof(real)*1);
               else
                 memcpy(dst+(size_t)(y*outputWidth+x), src+(size_t)(iy*inputWidth+ix), sizeof(real)*(1));
            }
          }
        }
      }
    } else {
      for(y = 0; y < outputHeight; y++) {
        iy = (long long)(y*dH + kh);
        ix = (long long)(0 + kw);
        if (dW == 1)
           memcpy(dst+(size_t)(y*outputWidth), src+(size_t)(iy*inputWidth+ix), sizeof(real)*outputWidth);
        else{
          for (x=0; x<outputWidth; x++)
             memcpy(dst+(size_t)(y*outputWidth+x), src+(size_t)(iy*inputWidth+ix+x*dW), sizeof(real)*(1));
         }
      }
    }
  }
}

#endif

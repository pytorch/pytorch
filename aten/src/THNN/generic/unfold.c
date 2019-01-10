#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/unfold.c"
#else

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
  // This function assumes that
  // outputHeight*dH does not overflow a int64_t
  // outputWidth*dW does not overflow a int64_t

  int nip;

  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(nip)
  for(nip = 0; nip < nInputPlane; nip++)
  {
    int kw, kh, y, x;
    int64_t ix, iy;
    for(kh = 0; kh < kH; kh++)
    {
      for(kw = 0; kw < kW; kw++)
      {
        real *src = finput_data + nip*((size_t)kH*kW*outputHeight*outputWidth) + kh*((size_t)kW*outputHeight*outputWidth) + kw*((size_t)outputHeight*outputWidth);
        real *dst = input_data + nip*((size_t)inputHeight*inputWidth);
        if (padW > 0 || padH > 0) {
          int lpad,rpad;
          for(y = 0; y < outputHeight; y++) {
            iy = (int64_t)y*dH - padH + kh;
            if (iy < 0 || iy >= inputHeight) {
            } else {
              if (dW==1){
                 ix = 0 - padW + kw;
                 lpad = fmaxf(0,padW-kw);
                 rpad = fmaxf(0,padW-(kW-kw-1));
                 real *dst_slice = dst+(size_t)iy*inputWidth+ix+lpad;
                 THVector_(cadd)(dst_slice, dst_slice, src+(size_t)y*outputWidth+lpad, 1, outputWidth - lpad - rpad); /* note: THVector_add could handle 1 value better */
              }
              else{
                for (x=0; x<outputWidth; x++){
                   ix = (int64_t)x*dW - padW + kw;
                   if (ix < 0 || ix >= inputWidth){
                   }else{
                     real *dst_slice = dst+(size_t)iy*inputWidth+ix;
                     THVector_(cadd)(dst_slice, dst_slice, src+(size_t)y*outputWidth+x, 1, 1);
                   }
                }
              }
            }
          }
        } else {
          for(y = 0; y < outputHeight; y++) {
            iy = (int64_t)y*dH + kh;
            ix = 0 + kw;
            if (dW == 1 ) {
               real *dst_slice = dst+(size_t)iy*inputWidth+ix;
               THVector_(cadd)(dst_slice, dst_slice, src+(size_t)y*outputWidth, 1, outputWidth); /* note: THVector_add could handle 1 value better */
            }else{
              for(x = 0; x < outputWidth; x++) {
                real *dst_slice = dst+(size_t)iy*inputWidth+ix+x*dW;
                THVector_(cadd)(dst_slice, dst_slice, src+(size_t)y*outputWidth+x, 1, 1);
              }
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
  // This function assumes that
  // kH*kW does not overflow an int
  // nInputPlane*kH*kW does not overflow a int64_t
  // outputHeight*dH does not overflow a int64_t
  // outputWidth*dW does not overflow a int64_t

  int64_t k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(k)
  for(k = 0; k < (int64_t)nInputPlane*kH*kW; k++) {
    int64_t nip = k / (kH*kW);
    int64_t rest = k % (kH*kW);
    int64_t kh = rest / kW;
    int64_t kw = rest % kW;
    int x, y;
    int64_t ix, iy;
    real *dst = finput_data + nip*((size_t)kH*kW*outputHeight*outputWidth) + kh*((size_t)kW*outputHeight*outputWidth) + kw*((size_t)outputHeight*outputWidth);
    real *src = input_data + nip*((size_t)inputHeight*inputWidth);
    if (padW > 0 || padH > 0) {
      int64_t lpad,rpad;
      for(y = 0; y < outputHeight; y++) {
        iy = (int64_t)y*dH - padH + kh;
        if (iy < 0 || iy >= inputHeight) {
          memset(dst+(size_t)y*outputWidth, 0, sizeof(real)*outputWidth);
        } else {
          if (dW==1){
             ix = 0 - padW + kw;
             lpad = fmaxf(0,padW-kw);
             rpad = fmaxf(0,padW-(kW-kw-1));
             if (outputWidth-rpad-lpad <= 0) {
                memset(dst+(size_t)y*outputWidth, 0, sizeof(real)*outputWidth);
             } else {
                if (lpad > 0) memset(dst+(size_t)y*outputWidth, 0, sizeof(real)*lpad);
                memcpy(dst+(size_t)y*outputWidth+lpad, src+(size_t)iy*inputWidth+ix+lpad, sizeof(real)*(outputWidth-rpad-lpad));
                if (rpad > 0) memset(dst+(size_t)y*outputWidth + outputWidth - rpad, 0, sizeof(real)*rpad);
             }
          }
          else{
            for (x=0; x<outputWidth; x++){
               ix = (int64_t)x*dW - padW + kw;
               if (ix < 0 || ix >= inputWidth)
                 memset(dst+(size_t)y*outputWidth+x, 0, sizeof(real)*1);
               else
                 memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix, sizeof(real)*(1));
            }
          }
        }
      }
    } else {
      for(y = 0; y < outputHeight; y++) {
        iy = (int64_t)y*dH + kh;
        ix = 0 + kw;
        if (dW == 1)
           memcpy(dst+(size_t)y*outputWidth, src+(size_t)iy*inputWidth+ix, sizeof(real)*outputWidth);
        else{
          for (x=0; x<outputWidth; x++)
             memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix+(int64_t)x*dW, sizeof(real)*(1));
         }
      }
    }
  }
}

#endif

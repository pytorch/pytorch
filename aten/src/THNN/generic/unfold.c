#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/unfold.c"
#else

#include <ATen/Parallel.h>

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

  scalar_t *input_data = input->data<scalar_t>();
  scalar_t *finput_data = finput->data<scalar_t>();

  at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
    for (auto nip = start; nip < end; nip++)
    {
      int kw, kh, y, x;
      int64_t ix, iy;
      for(kh = 0; kh < kH; kh++)
      {
        for(kw = 0; kw < kW; kw++)
        {
          scalar_t *src = finput_data + nip*((size_t)kH*kW*outputHeight*outputWidth) + kh*((size_t)kW*outputHeight*outputWidth) + kw*((size_t)outputHeight*outputWidth);
          scalar_t *dst = input_data + nip*((size_t)inputHeight*inputWidth);
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
                   scalar_t *dst_slice = dst+(size_t)iy*inputWidth+ix+lpad;
                   THVector_(cadd)(dst_slice, dst_slice, src+(size_t)y*outputWidth+lpad, 1, outputWidth - lpad - rpad); /* note: THVector_add could handle 1 value better */
                }
                else{
                  for (x=0; x<outputWidth; x++){
                     ix = (int64_t)x*dW - padW + kw;
                     if (ix < 0 || ix >= inputWidth){
                     }else{
                       scalar_t *dst_slice = dst+(size_t)iy*inputWidth+ix;
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
                 scalar_t *dst_slice = dst+(size_t)iy*inputWidth+ix;
                 THVector_(cadd)(dst_slice, dst_slice, src+(size_t)y*outputWidth, 1, outputWidth); /* note: THVector_add could handle 1 value better */
              }else{
                for(x = 0; x < outputWidth; x++) {
                  scalar_t *dst_slice = dst+(size_t)iy*inputWidth+ix+x*dW;
                  THVector_(cadd)(dst_slice, dst_slice, src+(size_t)y*outputWidth+x, 1, 1);
                }
              }
            }
          }
        }
      }
    }
  });
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

  scalar_t *input_data = input->data<scalar_t>();
  scalar_t *finput_data = finput->data<scalar_t>();

  at::parallel_for(0, (int64_t)nInputPlane*kH*kW, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++) {
      int64_t nip = k / (kH*kW);
      int64_t rest = k % (kH*kW);
      int64_t kh = rest / kW;
      int64_t kw = rest % kW;
      int x, y;
      int64_t ix, iy;
      scalar_t *dst = finput_data + nip*((size_t)kH*kW*outputHeight*outputWidth) + kh*((size_t)kW*outputHeight*outputWidth) + kw*((size_t)outputHeight*outputWidth);
      scalar_t *src = input_data + nip*((size_t)inputHeight*inputWidth);
      if (padW > 0 || padH > 0) {
        int64_t lpad,rpad;
        for(y = 0; y < outputHeight; y++) {
          iy = (int64_t)y*dH - padH + kh;
          if (iy < 0 || iy >= inputHeight) {
            memset(dst+(size_t)y*outputWidth, 0, sizeof(scalar_t)*outputWidth);
          } else {
            if (dW==1){
               ix = 0 - padW + kw;
               lpad = fmaxf(0,padW-kw);
               rpad = fmaxf(0,padW-(kW-kw-1));
               if (outputWidth-rpad-lpad <= 0) {
                  memset(dst+(size_t)y*outputWidth, 0, sizeof(scalar_t)*outputWidth);
               } else {
                  if (lpad > 0) memset(dst+(size_t)y*outputWidth, 0, sizeof(scalar_t)*lpad);
                  memcpy(dst+(size_t)y*outputWidth+lpad, src+(size_t)iy*inputWidth+ix+lpad, sizeof(scalar_t)*(outputWidth-rpad-lpad));
                  if (rpad > 0) memset(dst+(size_t)y*outputWidth + outputWidth - rpad, 0, sizeof(scalar_t)*rpad);
               }
            }
            else{
              for (x=0; x<outputWidth; x++){
                 ix = (int64_t)x*dW - padW + kw;
                 if (ix < 0 || ix >= inputWidth)
                   memset(dst+(size_t)y*outputWidth+x, 0, sizeof(scalar_t)*1);
                 else
                   memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix, sizeof(scalar_t)*(1));
              }
            }
          }
        }
      } else {
        for(y = 0; y < outputHeight; y++) {
          iy = (int64_t)y*dH + kh;
          ix = 0 + kw;
          if (dW == 1)
             memcpy(dst+(size_t)y*outputWidth, src+(size_t)iy*inputWidth+ix, sizeof(scalar_t)*outputWidth);
          else{
            for (x=0; x<outputWidth; x++)
               memcpy(dst+(size_t)y*outputWidth+x, src+(size_t)iy*inputWidth+ix+(int64_t)x*dW, sizeof(scalar_t)*(1));
           }
        }
      }
    }
  });
}

#endif

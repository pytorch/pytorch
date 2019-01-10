#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorConv.cpp"
#else

/*
  2D Input, 2D kernel  : convolve given image with the given kernel.
*/
void THTensor_(validXCorr2Dptr)(real *r_,
                                       real alpha,
                                       real *t_, int64_t ir, int64_t ic,
                                       real *k_, int64_t kr, int64_t kc,
                                       int64_t sr, int64_t sc)
{
  int64_t or_ = (ir - kr) / sr + 1;
  int64_t oc = (ic - kc) / sc + 1;

  int64_t xx, yy, kx, ky;

  if ((sc != 1) || (oc < 4))  {
    /* regular convolution */
    for(yy = 0; yy < or_; yy++) {
      for(xx = 0; xx < oc; xx++) {
        /* Dot product in two dimensions... (between input image and the mask) */
        real *pi_ = t_ + yy*sr*ic + xx*sc;
        real *pw_ = k_;
        real sum = 0;
        for(ky = 0; ky < kr; ky++) {
          for(kx = 0; kx < kc; kx++) {
            sum += pi_[kx]*pw_[kx];
          }
          pi_ += ic; /* next input line */
          pw_ += kc; /* next mask line */
        }
        /* Update output */
        *r_++ += alpha*sum;
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < or_; yy++) {
      real *pi_ = t_ + yy*sr*ic;
      real *pw_ = k_;
      for (ky = 0; ky < kr; ky++) {
        real *pis_ = pi_;
        for (kx = 0; kx < kc; kx++) {
          THVector_(cadd)(r_, r_, pis_, alpha*pw_[kx], oc);
          pis_++;
        }
        pi_ += ic; /* next input line */
        pw_ += kc; /* next mask line */
      }
      r_ += oc;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel.
*/
void THTensor_(validConv2Dptr)(real *r_,
                                      real alpha,
                                      real *t_, int64_t ir, int64_t ic,
                                      real *k_, int64_t kr, int64_t kc,
                                      int64_t sr, int64_t sc)
{
  int64_t or_ = (ir - kr) / sr + 1;
  int64_t oc = (ic - kc) / sc + 1;

  int64_t xx, yy, kx, ky;

  if ((sc != 1) || (oc < 4))  {
    /* regular convolution */
    for(yy = 0; yy < or_; yy++) {
      for(xx = 0; xx < oc; xx++) {
        /* Dot product in two dimensions... (between input image and the mask) */
        real *pi_ = t_ + yy*sr*ic + xx*sc;
        real *pw_ = k_ + kr*kc - 1;
        real sum = 0;
        for(ky = 0; ky < kr; ky++) {
          for(kx = 0; kx < kc; kx++) {
            sum += pi_[kx]*pw_[-kx];
          }
          pi_ += ic; /* next input line */
          pw_ -= kc; /* next mask line */
        }
        /* Update output */
        *r_++ += alpha*sum;
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < or_; yy++) {
      real *pw_ = k_ + kr*kc - 1;
      real *pi_ = t_ + yy*sr*ic;
      for (ky = 0; ky < kr; ky++) {
        real *pis_ = pi_;
        for (kx = 0; kx < kc; kx++) {
          THVector_(cadd)(r_, r_, pis_, alpha*pw_[-kx], oc);
          pis_++;
        }
        pi_ += ic; /* next input line */
        pw_ -= kc; /* next mask line */
      }
      r_ += oc;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel, full convolution.
*/
void THTensor_(fullConv2Dptr)(real *r_,
                                     real alpha,
                                     real *t_, int64_t ir, int64_t ic,
                                     real *k_, int64_t kr, int64_t kc,
                                     int64_t sr, int64_t sc)
{
  int64_t oc = (ic - 1) * sc + kc;

  int64_t xx, yy, kx, ky;

  if ((sc != 1) || (ic < 4))  {
    /* regular convolution */
    for(yy = 0; yy < ir; yy++) {
      for(xx = 0; xx < ic; xx++) {
        /* Outer product in two dimensions... (between input image and the mask) */
        real *po_ = r_ + yy*sr*oc + xx*sc;
        real *pw_ = k_;
        for(ky = 0; ky < kr; ky++)
        {
          real z = *t_ * alpha;
          for(kx = 0; kx < kc; kx++) {
            po_[kx] += z * pw_[kx];
          }
          po_ += oc; /* next input line */
          pw_ += kc; /* next mask line */
        }
        t_++;
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < ir; yy++) {
      real *po_ = r_ + yy*sr*oc;
      real *pw_ = k_;
      for (ky = 0; ky < kr; ky++) {
        real *pos_ = po_;
        for (kx = 0; kx < kc; kx++) {
          THVector_(cadd)(pos_, pos_, t_, alpha*pw_[kx], ic);
          pos_++;
        }
        po_ += oc; /* next input line */
        pw_ += kc; /* next mask line */
      }
      t_ += ic;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel, full convolution.
*/
void THTensor_(fullXCorr2Dptr)(real *r_,
                                      real alpha,
                                      real *t_, int64_t ir, int64_t ic,
                                      real *k_, int64_t kr, int64_t kc,
                                      int64_t sr, int64_t sc)
{
  int64_t oc = (ic - 1) * sc + kc;

  int64_t xx, yy, kx, ky;

  if ((sc != 1) || (ic < 4))  {
    /* regular convolution */
    for(yy = 0; yy < ir; yy++) {
      for(xx = 0; xx < ic; xx++) {
        /* Outer product in two dimensions... (between input image and the mask) */
        real *po_ = r_ + yy*sr*oc + xx*sc;
        real *pw_ = k_ + kr*kc -1;
        int64_t kx, ky;
        for(ky = 0; ky < kr; ky++)
        {
          real z = *t_ * alpha;
          for(kx = 0; kx < kc; kx++) {
            po_[kx] += z * pw_[-kx];
          }
          po_ += oc; /* next input line */
          pw_ -= kc; /* next mask line */
        }
        t_++;
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < ir; yy++) {
      real *po_ = r_ + yy*sr*oc;
      real *pw_ = k_ + kr*kc -1;
      for (ky = 0; ky < kr; ky++) {
        real *pos_ = po_;
        for (kx = 0; kx < kc; kx++) {
          THVector_(cadd)(pos_, pos_, t_, pw_[-kx]*alpha, ic);
          pos_++;
        }
        po_ += oc; /* next input line */
        pw_ -= kc; /* next mask line */
      }
      t_ += ic;
    }
  }
}

/*
  2D Input, 2D kernel  : convolve given image with the given kernel, valid convolution.
  for sr,sc=1 this is equivalent to validXCorr2Dptr, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
void THTensor_(validXCorr2DRevptr)(real *r_,
                                          real alpha,
                                          real *t_, int64_t ir, int64_t ic,
                                          real *k_, int64_t kr, int64_t kc,
                                          int64_t sr, int64_t sc)
{
  int64_t or_ = ir - (kr - 1) * sr;
  int64_t oc = ic - (kc - 1) * sc;

  int64_t xx, yy, kx, ky;

  if ((sc != 1) || (kc < 4))  {
    /* regular convolution */
    for(yy = 0; yy < kr; yy++) {
      for(xx = 0; xx < kc; xx++) {
        real *po_ = r_;
        real *pi_ = t_ + yy*sr*ic + xx*sc;
        real z = *k_++ * alpha;

        for(ky = 0; ky < or_; ky++) {
          for(kx = 0; kx < oc; kx++)
            po_[kx] += z * pi_[kx];
          pi_ += ic;
          po_ += oc;
        }
      }
    }

  } else {
    /* SSE-based convolution */
    for(yy = 0; yy < kr; yy++) {
      for(xx = 0; xx < kc; xx++) {
        real *po_ = r_;
        real *pi_ = t_ + yy*sr*ic + xx*sc;
        real z = *k_++ * alpha;

        for(ky = 0; ky < or_; ky++) {
          THVector_(cadd)(po_, po_, pi_, z, oc);
          pi_ += ic;
          po_ += oc;
        }
      }
    }
  }
}
/*
  3D Input, 3D kernel  : convolve given volume with the given kernel.
*/
void THTensor_(validXCorr3Dptr)(real *r_,
                                       real alpha,
                                       real *t_, int64_t it, int64_t ir, int64_t ic,
                                       real *k_, int64_t kt, int64_t kr, int64_t kc,
                                       int64_t st, int64_t sr, int64_t sc)
{
  int64_t ot = (it - kt) / st + 1;
  int64_t or_ = (ir - kr) / sr + 1;
  int64_t oc = (ic - kc) / sc + 1;

  int64_t zz, xx, yy;

  for (zz = 0; zz < ot; zz++)
  {
    for(yy = 0; yy < or_; yy++)
    {
      for(xx = 0; xx < oc; xx++)
      {
        /* Dot product in two dimensions... (between input image and the mask) */
        real *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
        real *pw_ = k_;
        real sum = 0;
        int64_t kz, kx, ky;
        for(kz = 0; kz < kt; kz++)
        {
          for(ky = 0; ky < kr; ky++)
          {
            for(kx = 0; kx < kc; kx++) {
              sum += pi_[kx]*pw_[kx];
            }
            pi_ += ic; /* next input line */
            pw_ += kc; /* next mask line */
          }
          pi_ += (ir-kr)*ic; /* next input slice */
        }
        /* Update output */
        *r_++ += sum*alpha;
      }
    }
  }
}

/*
  3D Input, 3D kernel  : convolve given volume with the given kernel.
*/
void THTensor_(validConv3Dptr)(real *r_,
                                      real alpha,
                                      real *t_, int64_t it, int64_t ir, int64_t ic,
                                      real *k_, int64_t kt, int64_t kr, int64_t kc,
                                      int64_t st, int64_t sr, int64_t sc)
{
  int64_t ot = (it - kt) / st + 1;
  int64_t or_ = (ir - kr) / sr + 1;
  int64_t oc = (ic - kc) / sc + 1;

  int64_t zz, xx, yy;

  for(zz = 0; zz < ot; zz++)
  {
    for(yy = 0; yy < or_; yy++)
    {
      for(xx = 0; xx < oc; xx++)
      {
        /* Dot product in two dimensions... (between input image and the mask) */
        real *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
        real *pw_ = k_ + kt*kr*kc - 1;
        real sum = 0;
        int64_t kz, kx, ky;
        for(kz = 0; kz < kt; kz++)
        {
          for(ky = 0; ky < kr; ky++)
          {
            for(kx = 0; kx < kc; kx++) {
              sum += pi_[kx]*pw_[-kx];
            }
            pi_ += ic; /* next input line */
            pw_ -= kc; /* next mask line */
          }
          pi_ += (ir-kr)*ic; /* next input slice */
        }
        /* Update output */
        *r_++ += alpha*sum;
      }
    }
  }
}


/*
  3D Input, 3D kernel  : convolve given volume with the given kernel, full convolution.
*/
void THTensor_(fullConv3Dptr)(real *r_,
                                     real alpha,
                                     real *t_, int64_t it, int64_t ir, int64_t ic,
                                     real *k_, int64_t kt, int64_t kr, int64_t kc,
                                     int64_t st, int64_t sr, int64_t sc)
{
  int64_t or_ = (ir - 1) * sr + kr;
  int64_t oc = (ic - 1) * sc + kc;

  int64_t zz, xx, yy;

  for(zz = 0; zz < it; zz++)
  {
    for(yy = 0; yy < ir; yy++)
    {
      for(xx = 0; xx < ic; xx++)
      {
        /* Outer product in two dimensions... (between input image and the mask) */
        real *po_ = r_ + zz*st*or_*oc + yy*sr*oc + xx*sc;
        real *pw_ = k_;
        int64_t kz, kx, ky;
        /* printf("Output Plane : %ld,%ld,%ld, input val=%g\n",zz,yy,xx,*t_); */
        for(kz = 0; kz < kt; kz++)
        {
          for(ky = 0; ky < kr; ky++)
          {
            real z = *t_ * alpha;
            for(kx = 0; kx < kc; kx++) {
              /* printf("o=%g,k=%g," , po_[kx],pw_[kx]); */
              po_[kx] += z * pw_[kx];
              /* printf("o=%g " , po_[kx]); */
            }
            /* printf("\n"); */
            po_ += oc; /* next input line */
            pw_ += kc; /* next mask line */
          }
          po_ += (or_-kr)*oc; /* next output slice */
          /* printf("\n"); */
        }
        t_++;
      }
    }
  }
}

/*
  3D Input, 3D kernel  : convolve given volume with the given kernel, full convolution.
*/
void THTensor_(fullXCorr3Dptr)(real *r_,
                                      real alpha,
                                      real *t_, int64_t it, int64_t ir, int64_t ic,
                                      real *k_, int64_t kt, int64_t kr, int64_t kc,
                                      int64_t st, int64_t sr, int64_t sc)
{
  int64_t or_ = (ir - 1) * sr + kr;
  int64_t oc = (ic - 1) * sc + kc;

  int64_t zz, xx, yy;

  for(zz = 0; zz < it; zz++)
  {
    for(yy = 0; yy < ir; yy++)
    {
      for(xx = 0; xx < ic; xx++)
      {
        /* Outer product in two dimensions... (between input image and the mask) */
        real *po_ = r_ + zz*st*or_*oc + yy*sr*oc + xx*sc;
        real *pw_ = k_ + kt*kr*kc -1;
        int64_t kz, kx, ky;
        for(kz = 0; kz < kt; kz++)
        {
          for(ky = 0; ky < kr; ky++)
          {
            real z = *t_ * alpha;
            for(kx = 0; kx < kc; kx++) {
              po_[kx] += z * pw_[-kx];
            }
            po_ += oc; /* next input line */
            pw_ -= kc; /* next mask line */
          }
          po_ += (or_-kr)*oc; /* next output slice */
        }
        t_++;
      }
    }
  }
}

/*
  3D Input, 3D kernel  : convolve given image with the given kernel, valid convolution.
  for sr,sc=1 this is equivalent to validXCorr3Dptr, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
void THTensor_(validXCorr3DRevptr)(real *r_,
                                          real alpha,
                                          real *t_, int64_t it, int64_t ir, int64_t ic,
                                          real *k_, int64_t kt, int64_t kr, int64_t kc,
                                          int64_t st, int64_t sr, int64_t sc)
{
  int64_t ot = it - (kt - 1) * st;
  int64_t or_ = ir - (kr - 1) * sr;
  int64_t oc = ic - (kc - 1) * sc;

  int64_t zz, xx, yy;
  for(zz = 0; zz < kt; zz++)
  {
    for(yy = 0; yy < kr; yy++)
    {
      for(xx = 0; xx < kc; xx++)
      {
        real *po_ = r_;
        real *pi_ = t_ + zz*st*ir*ic + yy*sr*ic + xx*sc;
        real z = *k_++ * alpha;
        int64_t kz, kx, ky;
        for(kz = 0; kz < ot; kz++)
        {
          for(ky = 0; ky < or_; ky++)
          {
            for(kx = 0; kx < oc; kx++)
              po_[kx] += z * pi_[kx];
            pi_ += ic;
            po_ += oc;
          }
          pi_ += (ir-or_)*ic; /* next input slice */
        }
      }
    }
  }
}

void THTensor_(conv2d)(real* output_data,
                       real alpha,
                       real* ptr_input, int64_t nInputRows, int64_t nInputCols,
                       real* ptr_weight, int64_t nKernelRows, int64_t nKernelCols,
                       int64_t srow, int64_t scol,
                       const char *vf, const char *xc)
{
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can be 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can be 'X' or 'C'");
  if (*vf == 'F')
    if (*xc == 'X')
      THTensor_(fullXCorr2Dptr)(output_data,
                                alpha,
                                ptr_input,  nInputRows,  nInputCols,
                                ptr_weight, nKernelRows, nKernelCols,
                                srow, scol);
    else
      THTensor_(fullConv2Dptr)(output_data,
                               alpha,
                               ptr_input,  nInputRows,  nInputCols,
                               ptr_weight, nKernelRows, nKernelCols,
                               srow, scol);
  else
    if (*xc == 'X')
      THTensor_(validXCorr2Dptr)(output_data,
                                 alpha,
                                 ptr_input,  nInputRows,  nInputCols,
                                 ptr_weight, nKernelRows, nKernelCols,
                                 srow, scol);
    else
      THTensor_(validConv2Dptr)(output_data,
                                alpha,
                                ptr_input,  nInputRows,  nInputCols,
                                ptr_weight, nKernelRows, nKernelCols,
                                srow, scol);
}

void THTensor_(conv3d)(real* output_data,
                       real alpha,
                       real* ptr_input, int64_t nInputDepth, int64_t nInputRows, int64_t nInputCols,
                       real* ptr_weight, int64_t nKernelDepth, int64_t nKernelRows, int64_t nKernelCols,
                       int64_t sdepth, int64_t srow, int64_t scol,
                       const char *vf, const char *xc)
{
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can be 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can be 'X' or 'C'");
  if (*vf == 'F')
    if (*xc == 'X')
      THTensor_(fullXCorr3Dptr)(output_data,
                                alpha,
                                ptr_input, nInputDepth, nInputRows,  nInputCols,
                                ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                sdepth, srow, scol);
    else
      THTensor_(fullConv3Dptr)(output_data,
                               alpha,
                               ptr_input, nInputDepth, nInputRows,  nInputCols,
                               ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                               sdepth, srow, scol);
  else
    if (*xc == 'X')
      THTensor_(validXCorr3Dptr)(output_data,
                                 alpha,
                                 ptr_input, nInputDepth, nInputRows,  nInputCols,
                                 ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                 sdepth, srow, scol);
    else
      THTensor_(validConv3Dptr)(output_data,
                                alpha,
                                ptr_input, nInputDepth, nInputRows,  nInputCols,
                                ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                sdepth, srow, scol);
}

int64_t THTensor_(convsize)(int64_t x, int64_t k, int64_t s, const char* vf)
{
  THArgCheck(*vf == 'V' || *vf == 'F', 1, "type of convolution can be 'V' or 'F'");
  if (*vf == 'V')
    return (x-k)/s + 1;
  else
    return (x-1)*s + k;
}


/*
  3D input, 3D kernel, 4D output
  like rank1 update
  A <- xx' + beta*A
  for sr,sc=1 this is equivalent to conv2Dger, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
void THTensor_(conv2DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol)
{
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelPlane, nKernelRows, nKernelCols;
  int64_t nOutputRows, nOutputCols;
  int64_t istride0, kstride0;
  THTensor *input;
  THTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0 = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "covn2DRevger : Input image is smaller than kernel");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_,nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    /*THTensor_(zero)(r_);*/

#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      int64_t l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }
  }
  else if (beta != 1)
  {
    /*THTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      int64_t l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] *= beta;
    }
  }

#pragma omp parallel for private(k)
  for(k = 0; k < nKernelPlane; k++)
  {
    int64_t i;
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get output */
      real *ptr_output = output_data + k*nInputPlane*nOutputCols*nOutputRows + i*nOutputCols*nOutputRows;
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      THTensor_(validXCorr2DRevptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows; */
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}


/*
  3D input, 3D kernel, 4D output
  like rank1 update
  A <- xx' + beta*A
  for sr,sc=1 this is equivalent to conv2Dger, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
void THTensor_(conv2DRevgerm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol)
{
  int64_t nbatch, nInputPlane, nInputRows, nInputCols;
  int64_t nKernelPlane, nKernelRows, nKernelCols;
  int64_t nOutputRows, nOutputCols;
  int64_t istride0, kstride0, istride1, kstride1;
  THTensor *input;
  THTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  istride1    = input->stride[1];
  nbatch      = input->size[0];
  nInputPlane = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0 = kernel->stride[0];
  kstride1 = kernel->stride[1];
  nKernelPlane = kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "conv2DRevger : Input image is smaller than kernel");
  THArgCheck(kernel->size[0] == input->size[0] , 2, "conv2DRevger : Input batch and kernel batch is not same size");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_,nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    /*THTensor_(zero)(r_);*/

#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      int64_t l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }
  }
  else if (beta != 1)
  {
    /*THTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      int64_t l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] *= beta;
    }
  }

#pragma omp parallel for private(k)
  for(k = 0; k < nKernelPlane; k++)
  {
    int64_t i;
    for(i = 0; i < nInputPlane; i++)
    {
      int64_t p;
      for(p = 0; p < nbatch; p++)
      {
        /* get kernel */
        real *ptr_weight = weight_data + p*kstride0 + k*kstride1;
        /* get output */
        real *ptr_output = output_data + k*nInputPlane*nOutputCols*nOutputRows + i*nOutputCols*nOutputRows;
        /* get input */
        real *ptr_input = input_data + p*istride0 + i*istride1;

        /* do image, kernel convolution */
        THTensor_(validXCorr2DRevptr)(ptr_output,
                                      alpha,
                                      ptr_input,  nInputRows,  nInputCols,
                                      ptr_weight, nKernelRows, nKernelCols,
                                      srow, scol);
        /* Next output plane */
        /* output_data += nOutputCols*nOutputRows; */
      }
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}


/*
  3D input, 3D kernel, 4D output
  like rank1 update
  A <- xx' + beta*A
*/
void THTensor_(conv2Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelPlane, nKernelRows, nKernelCols;
  int64_t nOutputRows, nOutputCols;
  int64_t istride0, kstride0;

  THTensor *input;
  THTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can 'X' or 'C'");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0 = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dger : Input image is smaller than kernel");

  if (*vf == 'F') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { /* valid */
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_, nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    /*THTensor_(zero)(r_);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      int64_t l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }
  }
  else if (beta != 1)
  {
    /*THTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]*r_->size[1]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      int64_t l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] *= beta;
    }
  }

#pragma omp parallel for private(k)
  for(k = 0; k < nKernelPlane; k++)
  {
    int64_t i;
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get output */
      real *ptr_output = output_data + k*nInputPlane*nOutputCols*nOutputRows + i*nOutputCols*nOutputRows;
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      if (*vf == 'F')
        if (*xc == 'X')
          THTensor_(fullXCorr2Dptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
        else
          THTensor_(fullConv2Dptr)(ptr_output,
                                   alpha,
                                   ptr_input,  nInputRows,  nInputCols,
                                   ptr_weight, nKernelRows, nKernelCols,
                                   srow, scol);
      else
        if (*xc == 'X')
          THTensor_(validXCorr2Dptr)(ptr_output,
                                     alpha,
                                     ptr_input,  nInputRows,  nInputCols,
                                     ptr_weight, nKernelRows, nKernelCols,
                                     srow, scol);
        else
          THTensor_(validConv2Dptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows; */
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}


/*
  3D input, 4D kernel, 3D output
  matrix vector product like
  y <- Ax + beta*y
*/
void THTensor_(conv2Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputRows, nOutputCols;
  int64_t istride0, kstride0, kstride1;
  THTensor *input;
  THTensor* kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can 'X' or 'C'");

  input = THTensor_(newContiguous)(t_);
  if (!(k_->stride[3] == 1) || !(k_->stride[2] == k_->size[3])) {
    kernel = THTensor_(newContiguous)(k_);
  } else {
    THTensor_(retain)(k_);
    kernel = k_;
  }

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0    = kernel->stride[0];
  kstride1    = kernel->stride[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dmv : Input image is smaller than kernel");

  if (*vf == 'F') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { /* valid */
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize3d)(r_, nOutputPlane, nOutputRows, nOutputCols);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    /*THTensor_(zero)(r_);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      int64_t l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }
  }
  else if (beta != 1)
  {
    /*THTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(k)
    for (k = 0; k < r_->size[0]; k++)
    {
      real* ptr_output = output_data + k*nOutputCols*nOutputRows;
      int64_t l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] *= beta;
    }
  }

#pragma omp parallel for private(k)
  for(k = 0; k < nOutputPlane; k++)
  {
    int64_t i;
    /* get output */
    real *ptr_output = output_data + k*nOutputCols*nOutputRows;
    for(i = 0; i < nInputPlane; i++)
    {
      /* get kernel */
      real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
      /* get input */
      real *ptr_input = input_data + i*istride0;

      /* do image, kernel convolution */
      if (*vf == 'F')
        if (*xc == 'X')
          THTensor_(fullXCorr2Dptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
        else
          THTensor_(fullConv2Dptr)(ptr_output,
                                   alpha,
                                   ptr_input,  nInputRows,  nInputCols,
                                   ptr_weight, nKernelRows, nKernelCols,
                                   srow, scol);
      else
        if (*xc == 'X')
          THTensor_(validXCorr2Dptr)(ptr_output,
                                     alpha,
                                     ptr_input,  nInputRows,  nInputCols,
                                     ptr_weight, nKernelRows, nKernelCols,
                                     srow, scol);
        else
          THTensor_(validConv2Dptr)(ptr_output,
                                    alpha,
                                    ptr_input,  nInputRows,  nInputCols,
                                    ptr_weight, nKernelRows, nKernelCols,
                                    srow, scol);
    }
    /* Next output plane */
    /* output_data += nOutputCols*nOutputRows;*/
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}


/*
  3D input, 4D kernel, 3D output
  matrix vector product like
  y <- Ax + beta*y
*/
void THTensor_(conv2Dmm)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputRows, nOutputCols;
  int64_t kstride0, kstride1;
  THTensor *input;
  THTensor* kernel;
  int64_t nbatch;
  ptrdiff_t nelem;
  real *input_data;
  real *weight_data;
  real *output_data;
  int64_t p;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can 'X' or 'C'");

  input = THTensor_(newContiguous)(t_);
  if (!(k_->stride[3] == 1) || !(k_->stride[2] == k_->size[3])) {
    kernel = THTensor_(newContiguous)(k_);
  } else {
    THTensor_(retain)(k_);
    kernel = k_;
  }

  nbatch = input->size[0];
  nInputPlane = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0    = kernel->stride[0];
  kstride1    = kernel->stride[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dmv : Input image is smaller than kernel");

  if (*vf == 'F') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { /* valid */
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_, nbatch, nOutputPlane, nOutputRows, nOutputCols);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    /*THTensor_(zero)(r_);*/
#pragma omp parallel for private(p)
    for (p=0; p < r_->size[0]; p++)
    {
      int64_t k;
      for (k = 0; k < r_->size[1]; k++)
      {
        real* ptr_output = output_data + p*nOutputPlane*nOutputRows*nOutputCols + k*nOutputCols*nOutputRows;
        int64_t l;
        for (l = 0; l < nOutputRows*nOutputCols; l++)
          ptr_output[l] = 0.0;
      }
    }
  }
  else if (beta != 1)
  {
    /*THTensor_(mul)(r_, beta);*/
#pragma omp parallel for private(p)
    for(p=0; p < r_->size[0]; p++)
    {
      int64_t k;
      for (k = 0; k < r_->size[1]; k++)
      {
        real* ptr_output = output_data + p*nOutputPlane*nOutputRows*nOutputCols + k*nOutputCols*nOutputRows;
        int64_t l;
        for (l = 0; l < nOutputRows*nOutputCols; l++)
          ptr_output[l] *= beta;
      }
    }
  }

#pragma omp parallel for private(p)
  for(p=0; p < nbatch; p++)
  {
    int64_t k;
    for(k = 0; k < nOutputPlane; k++)
    {
      int64_t i;
      /* get output */
      real *ptr_output = output_data + p*nOutputPlane*nOutputCols*nOutputRows + k*nOutputCols*nOutputRows;
      for(i = 0; i < nInputPlane; i++)
      {
        /* get kernel */
        real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
        /* get input */
        real *ptr_input = input_data + p*nInputPlane*nInputRows*nInputCols + i*nInputRows*nInputCols;

        /* do image, kernel convolution */
        if (*vf == 'F')
          if (*xc == 'X')
            THTensor_(fullXCorr2Dptr)(ptr_output,
                                      alpha,
                                      ptr_input,  nInputRows,  nInputCols,
                                      ptr_weight, nKernelRows, nKernelCols,
                                      srow, scol);
          else
            THTensor_(fullConv2Dptr)(ptr_output,
                                     alpha,
                                     ptr_input,  nInputRows,  nInputCols,
                                     ptr_weight, nKernelRows, nKernelCols,
                                     srow, scol);
        else
          if (*xc == 'X')
            THTensor_(validXCorr2Dptr)(ptr_output,
                                       alpha,
                                       ptr_input,  nInputRows,  nInputCols,
                                       ptr_weight, nKernelRows, nKernelCols,
                                       srow, scol);
          else
            THTensor_(validConv2Dptr)(ptr_output,
                                      alpha,
                                      ptr_input,  nInputRows,  nInputCols,
                                      ptr_weight, nKernelRows, nKernelCols,
                                      srow, scol);
      }
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows;*/
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}


/*
  2D input, 2D kernel, 2D output
  scalar multiplication like
  y <- x*y + beta*y
*/
void THTensor_(conv2Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  THTensor *input;
  THTensor* kernel;
  int64_t nInputRows;
  int64_t nInputCols;
  int64_t nKernelRows;
  int64_t nKernelCols;
  int64_t nOutputRows, nOutputCols;
  real *ptr_input;
  real *ptr_weight;
  real *output_data;
  ptrdiff_t nelem;

  THArgCheck(t_->nDimension == 2 , 3, "input: 2D Tensor expected");
  THArgCheck(k_->nDimension == 2 , 4, "kernel: 2D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  nInputRows  = input->size[0];
  nInputCols  = input->size[1];
  nKernelRows = kernel->size[0];
  nKernelCols = kernel->size[1];

  THArgCheck((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dmul : Input image is smaller than kernel");

  nOutputRows = THTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize2d)(r_, nOutputRows, nOutputCols);
  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
    THTensor_(zero)(r_);
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  ptr_input = THTensor_(data)(input);
  ptr_weight = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);


  /* do image, kernel convolution */
  THTensor_(conv2d)(output_data,
                    alpha,
                    ptr_input, nInputRows, nInputCols,
                    ptr_weight, nKernelRows, nKernelCols,
                    srow, scol, vf, xc);
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/*
  3D input, 3D kernel, 3D output
  component wise multiplication like
  y <- y.*x + beta*y
*/
void THTensor_(conv2Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputRows, nOutputCols;
  int64_t istride0, kstride0;
  THTensor *input;
  THTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0    = kernel->stride[0];
  nOutputPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck(nOutputPlane == nInputPlane, 2, "invalid number of input/kernel planes");
  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv2Dcmul : Input image is smaller than kernel");

  nOutputRows = THTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize3d)(r_, nOutputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  for(k = 0; k < nOutputPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data + k*kstride0;
    /* get input */
    real *ptr_input = input_data + k*istride0;

    /* do image, kernel convolution */
    THTensor_(conv2d)(output_data,
                      alpha,
                      ptr_input, nInputRows, nInputCols,
                      ptr_weight, nKernelRows, nKernelCols,
                      srow, scol, vf, xc);
    /* Next output plane */
    output_data += nOutputCols*nOutputRows;
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/*
  3D input, 3D kernel, 3D output
  component wise multiplication like with a permutation map
  y <- y.*x + beta*y
*/
void THTensor_(conv2Dmap)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, THTensor *map, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputRows, nOutputCols;
  int64_t istride0, kstride0;
  THTensor *input;
  THTensor* kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  int64_t nmaps;
  ptrdiff_t nelem;
  int64_t k;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(map->nDimension == 2 , 4, "map: 2D Tensor expected");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  kstride0    = kernel->stride[0];
  nOutputPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck(nOutputPlane == nInputPlane, 2, "invalid number of input/kernel planes");
  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols)
              || *vf == 'F', 2, "conv2Dmap : Input image is smaller than kernel");

  nOutputRows = THTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize3d)(r_, nOutputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  nmaps = map->size[0];

  for(k = 0; k < nmaps; k++)
  {
    /* get indices */
    int64_t from = (int64_t)THTensor_(get2d)(map,k,0)-1;
    int64_t to   = (int64_t)THTensor_(get2d)(map,k,1)-1;

    /* get kernel */
    real *ptr_weight = weight_data + k*kstride0;
    /* get input */
    real *ptr_input = input_data + from*istride0;
    /* get output */
    real *ptr_output = output_data + to*nOutputRows*nOutputCols;

    /* do image, kernel convolution */
    THTensor_(conv2d)(ptr_output,
                      alpha,
                      ptr_input, nInputRows, nInputCols,
                      ptr_weight, nKernelRows, nKernelCols,
                      srow, scol, vf, xc);
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/*
  4D input, 4D kernel, 5D output
  like rank1 update
  A <- xx' + beta*A
  for sr,sc=1 this is equivalent to xcorr2Dger, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
void THTensor_(conv3DRevger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_,
                             int64_t sdepth, int64_t srow, int64_t scol)
{
  int64_t nInputPlane, nInputDepth, nInputRows, nInputCols;
  int64_t nKernelPlane, nKernelDepth, nKernelRows, nKernelCols;
  int64_t nOutputDepth, nOutputRows, nOutputCols;
  int64_t istride0, kstride0;
  THTensor *input;
  THTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k, i;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(sdepth >= 1, 5, "Stride should be a positive integer");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0 = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelDepth= kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2, "conv3DRevger : Input image is smaller than kernel");

  nOutputDepth = nInputDepth - (nKernelDepth - 1) * sdepth;
  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize5d)(r_,nKernelPlane, nInputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  for(k = 0; k < nKernelPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      THTensor_(validXCorr3DRevptr)(output_data,
                                    alpha,
                                    ptr_input,  nInputDepth, nInputRows,  nInputCols,
                                    ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                                    sdepth, srow, scol);
      /* Next output plane */
      output_data += nOutputDepth*nOutputCols*nOutputRows;
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}


/*
  4D input, 4D kernel, 5D output
  like rank1 update
  A <- xx' + beta*A
*/
void THTensor_(conv3Dger)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_,
                          int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputDepth, nInputRows, nInputCols;
  int64_t nKernelPlane, nKernelDepth, nKernelRows, nKernelCols;
  int64_t nOutputDepth, nOutputRows, nOutputCols;
  int64_t istride0, kstride0;
  THTensor *input;
  THTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k, i;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(sdepth >= 1, 5, "Stride should be a positive integer");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 8, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 8, "type of convolution can 'X' or 'C'");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0     = kernel->stride[0];
  nKernelPlane = kernel->size[0];
  nKernelDepth = kernel->size[1];
  nKernelRows  = kernel->size[2];
  nKernelCols  = kernel->size[3];

  THArgCheck((nInputDepth >= nKernelDepth
              && nInputRows >= nKernelRows
              && nInputCols >= nKernelCols)
             || *vf == 'F', 2, "conv3Dger : Input image is smaller than kernel");

  nOutputDepth = THTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize5d)(r_,nKernelPlane, nInputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  for(k = 0; k < nKernelPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data+k*kstride0;

    for(i = 0; i < nInputPlane; i++)
    {
      /* get input */
      real *ptr_input = input_data+i*istride0;

      /* do image, kernel convolution */
      THTensor_(conv3d)(output_data,
                        alpha,
                        ptr_input,  nInputDepth, nInputRows,  nInputCols,
                        ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                        sdepth, srow, scol, vf, xc);

      /* Next output plane */
      output_data += nOutputDepth*nOutputCols*nOutputRows;
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/*
  4D input, 5D kernel, 4D output
  matrix vector product like
  y <- Ax + beta*y
*/
void THTensor_(conv3Dmv)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_,
                         int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputDepth, nInputRows, nInputCols;
  int64_t nKernelDepth, nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
  int64_t istride0, kstride0, kstride1;
  THTensor *input;
  THTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k, i;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 5 , 4, "kernel: 5D Tensor expected");
  THArgCheck(sdepth >= 1, 5, "Stride should be a positive integer");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 8, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 8, "type of convolution can 'X' or 'C'");

  input = THTensor_(newContiguous)(t_);
  if (!(k_->stride[4] == 1) || !(k_->stride[3] == k_->size[4])) {
    kernel = THTensor_(newContiguous)(k_);
  } else {
    THTensor_(retain)(k_);
    kernel = k_;
  }

  nInputPlane = input->size[0];
  istride0    = input->stride[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0    = kernel->stride[0];
  kstride1    = kernel->stride[1];
  nKernelDepth = kernel->size[2];
  nKernelRows = kernel->size[3];
  nKernelCols = kernel->size[4];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv3Dmv : Input image is smaller than kernel");

  nOutputDepth = THTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_, nOutputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  for(k = 0; k < nOutputPlane; k++)
  {
    for(i = 0; i < nInputPlane; i++)
    {
      /* get kernel */
      real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
      /* get input */
      real *ptr_input = input_data + i*istride0;

      /* do image, kernel convolution */
      THTensor_(conv3d)(output_data,
                        alpha,
                        ptr_input,  nInputDepth, nInputRows,  nInputCols,
                        ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                        sdepth, srow, scol, vf, xc);
    }
    /* Next output plane */
    output_data += nOutputDepth*nOutputCols*nOutputRows;
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/*
  3D input, 3D kernel, 3D output
  scalar multiplication like
  y <- x*y + beta*y
*/
void THTensor_(conv3Dmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_,
                          int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  THTensor *input;
  THTensor* kernel;
  int64_t nInputDepth;
  int64_t nInputRows;
  int64_t nInputCols;
  int64_t nKernelDepth;
  int64_t nKernelRows;
  int64_t nKernelCols;
  int64_t nOutputDepth, nOutputRows, nOutputCols;
  real *ptr_input;
  real *ptr_weight;
  real *output_data;
  ptrdiff_t nelem;

  THArgCheck(t_->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(sdepth >= 1, 5, "Stride should be a positive integer");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 8, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 8, "type of convolution can 'X' or 'C'");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  nInputDepth = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];
  nKernelDepth = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck((nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv3Dmul : Input image is smaller than kernel");

  nOutputDepth = THTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize3d)(r_, nOutputDepth, nOutputRows, nOutputCols);
  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
    THTensor_(zero)(r_);
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  ptr_input = THTensor_(data)(input);
  ptr_weight = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);


  /* do image, kernel convolution */
  THTensor_(conv3d)(output_data,
                    alpha,
                    ptr_input,  nInputDepth, nInputRows,  nInputCols,
                    ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                    sdepth, srow, scol, vf, xc);
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/*
  4D input, 4D kernel, 4D output
  component wise multiplication like
  y <- y.*x + beta*y
*/
void THTensor_(conv3Dcmul)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_,
                           int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputDepth, nInputRows, nInputCols;
  int64_t nKernelDepth, nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
  int64_t istride0, kstride0;

  THTensor *input;
  THTensor *kernel;
  real *input_data;
  real *weight_data;
  real *output_data;
  ptrdiff_t nelem;
  int64_t k;

  THArgCheck(t_->nDimension == 4 , 3, "input: 3D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 7, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 7, "type of convolution can 'X' or 'C'");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  nInputPlane = input->size[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0    = kernel->stride[0];
  nOutputPlane = kernel->size[0];
  nKernelDepth = kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nOutputPlane == nInputPlane, 2, "invalid number of input/kernel planes");
  THArgCheck( (nInputDepth >= nKernelDepth && nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *vf == 'F', 2, "conv3Dcmul : Input image is smaller than kernel");

  nOutputDepth = THTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_, nOutputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  for(k = 0; k < nOutputPlane; k++)
  {
    /* get kernel */
    real *ptr_weight = weight_data + k*kstride0;
    /* get input */
    real *ptr_input = input_data + k*istride0;

    /* do image, kernel convolution */
    THTensor_(conv3d)(output_data,
                      alpha,
                      ptr_input,  nInputDepth, nInputRows,  nInputCols,
                      ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                      sdepth, srow, scol, vf, xc);

    /* Next output plane */
    output_data += nOutputDepth*nOutputCols*nOutputRows;
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}

/*
  4D input, 4D kernel, 4D output
  component wise multiplication like with a permutation map
  y <- y.*x + beta*y
*/
void THTensor_(conv3Dmap)(THTensor *r_, real beta, real alpha, THTensor *t_, THTensor *k_, THTensor *map,
                          int64_t sdepth, int64_t srow, int64_t scol, const char *vf, const char *xc)
{
  int64_t nInputPlane, nInputDepth, nInputRows, nInputCols;
  int64_t nKernelDepth, nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputDepth, nOutputRows, nOutputCols;
  int64_t istride0, kstride0;

  THTensor *input;
  THTensor *kernel;
  ptrdiff_t nelem;
  real *input_data;
  real *weight_data;
  real *output_data;
  int64_t nmaps;
  int64_t k;

  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(map->nDimension == 2 , 4, "map: 2D Tensor expected");
  THArgCheck(srow >= 1, 6, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 7, "Stride should be a positive integer");
  THArgCheck(*vf == 'V' || *vf == 'F', 8, "type of convolution can 'V' or 'F'");
  THArgCheck(*xc == 'C' || *xc == 'X', 8, "type of convolution can 'X' or 'C'");

  input = THTensor_(newContiguous)(t_);
  kernel = THTensor_(newContiguous)(k_);

  istride0    = input->stride[0];
  nInputPlane = input->size[0];
  nInputDepth = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  kstride0    = kernel->stride[0];
  nOutputPlane = kernel->size[0];
  nKernelDepth = kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nOutputPlane == nInputPlane, 2, "invalid number of input/kernel planes");
  THArgCheck((nInputDepth >= nKernelDepth
              && nInputRows >= nKernelRows
              && nInputCols >= nKernelCols) || *vf == 'F',
             2, "conv3Dmap : Input image is smaller than kernel");

  nOutputDepth = THTensor_(convsize)(nInputDepth, nKernelDepth, sdepth, vf);
  nOutputRows = THTensor_(convsize)(nInputRows, nKernelRows, srow, vf);
  nOutputCols = THTensor_(convsize)(nInputCols, nKernelCols, scol, vf);

  nelem = THTensor_(nElement)(r_);
  THTensor_(resize4d)(r_, nOutputPlane, nOutputDepth, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
  {
    THTensor_(zero)(r_);
  }
  else if (beta != 1)
    THTensor_(mul)(r_, r_, beta);

  input_data = THTensor_(data)(input);
  weight_data = THTensor_(data)(kernel);
  output_data = THTensor_(data)(r_);

  nmaps = map->size[0];

  for(k = 0; k < nmaps; k++)
  {
    /* get indices */
    int64_t from = (int64_t)THTensor_(get2d)(map,k,0)-1;
    int64_t to   = (int64_t)THTensor_(get2d)(map,k,1)-1;

    /* get kernel */
    real *ptr_weight = weight_data + k*kstride0;
    /* get input */
    real *ptr_input = input_data + from*istride0;
    /* get output */
    real *ptr_output = output_data + to*nOutputDepth*nOutputRows*nOutputCols;

    /* do image, kernel convolution */
    THTensor_(conv3d)(ptr_output,
                      alpha,
                      ptr_input,  nInputDepth, nInputRows,  nInputCols,
                      ptr_weight, nKernelDepth, nKernelRows, nKernelCols,
                      sdepth, srow, scol, vf, xc);
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}
#endif

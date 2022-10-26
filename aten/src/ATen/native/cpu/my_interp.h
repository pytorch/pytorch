#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#include "resample_simd_horizontal.c"
#include "resample_simd_vertical.c"
#include "blah.h"  // TODO: remove
#include <stdio.h> // TODO: remove

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/ones.h>
#endif

#include <iostream>

namespace at {
namespace native {
namespace {



static inline double bilinear_filter(double x) {
    // TODO: Same as aa_filter()
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return 1.0 - x;
    }
    return 0.0;
}

void unpack_rgb(uint8_t * unpacked, const uint8_t * packed, int num_pixels)
// TODO: maybe use faster version from https://github.com/python-pillow/Pillow/pull/2693/files
{
    int i;
    for (i = 0; i < num_pixels; i++) {
        unpacked[0] = packed[0];
        unpacked[1] = packed[1];
        unpacked[2] = packed[2];
        unpacked[3] = 255;
        unpacked += 4;
        packed += 3;
    }
}

void pack_rgb(uint8_t * packed, const uint8_t* unpacked, int num_pixels)
// TODO: maybe use faster version from https://github.com/python-pillow/Pillow/pull/2693/files
{
    int i;
    /* RGB triplets */
    for (i = 0; i < num_pixels; i++) {
        packed [0] = unpacked[0];
        packed [1] = unpacked[1];
        packed [2] = unpacked[2];
        packed  += 3;
        unpacked+= 4;
    }
}

int precompute_coeffs( // TODO: This is redundant with `compute_indices_weights_aa()`
    int inSize,
    int outSize,
    int **boundsp,
    double **kkp) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    int *bounds;
    double *kk, *k;

    /* prepare for horizontal stretch */
    filterscale = scale = (double)inSize / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }
    /* determine support size (length of resampling filter) */
    // support = filterp->support * filterscale;
    support = filterscale; // bilinear filter support is 1 for pil-simd

    /* maximum number of coeffs */
    ksize = (int)ceil(support) * 2 + 1;

    // check for overflow
    if (outSize > INT_MAX / (ksize * (int)sizeof(double))) {
        // ImagingError_MemoryError();
        AT_ERROR("BLOP") ;
        return 0;
    }

    /* coefficient buffer */
    /* malloc check ok, overflow checked above */
    kk = (double *)malloc(outSize * ksize * sizeof(double));
    if (!kk) {
        // ImagingError_MemoryError();
        AT_ERROR("BLIP") ;
        return 0;
    }

    /* malloc check ok, ksize*sizeof(double) > 2*sizeof(int) */
    bounds = (int*)malloc(outSize * 2 * sizeof(int));
    if (!bounds) {
        free(kk);
        // ImagingError_MemoryError();
        AT_ERROR("BLOOP") ;
        return 0;
    }

    for (xx = 0; xx < outSize; xx++) {
        center = (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        // Round the value
        xmin = (int)(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        // Round the value
        xmax = (int)(center + support + 0.5);
        if (xmax > inSize) {
            xmax = inSize;
        }
        xmax -= xmin;
        k = &kk[xx * ksize];
        for (x = 0; x < xmax; x++) {
            double w = bilinear_filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0) {
                k[x] /= ww;
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < ksize; x++) {
            k[x] = 0;
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    *boundsp = bounds;
    *kkp = kk;
    return ksize;
}

#define PRECISION_BITS (32 - 8 - 2)
#define MAX_COEFS_PRECISION (16 - 1)

int normalize_coeffs_8bpc(int outSize, int ksize, double *prekk) {
    int x;
    INT16 *kk;
    int coefs_precision;
    double maxkk;

    // use the same buffer for normalized coefficients
    kk = (INT16 *) prekk;

    maxkk = prekk[0];
    for (x = 0; x < outSize * ksize; x++) {
        if (maxkk < prekk[x]) {
            maxkk = prekk[x];
        }
    }

    for (coefs_precision = 0; coefs_precision < PRECISION_BITS; coefs_precision += 1) {
        int next_value = (int) (0.5 + maxkk * (1 << (coefs_precision + 1)));
        // The next value will be outside of the range, so just stop
        if (next_value >= (1 << MAX_COEFS_PRECISION))
            break;
    }

    for (x = 0; x < outSize * ksize; x++) {
        if (prekk[x] < 0) {
            kk[x] = (int) (-0.5 + prekk[x] * (1 << coefs_precision));
        } else {
            kk[x] = (int) (0.5 + prekk[x] * (1 << coefs_precision));
        }
    }
    return coefs_precision;
}


 

void ImagingResampleHorizontal_8bpc(UINT32 * unpacked_output_p , UINT32 * unpacked_input_p, int offset, int ksize, int *bounds, double *prekk, int xout, int yout, int xin) {
    int ss0;
    int xx, yy, x, xmin, xmax;
    INT16 *k, *kk;
    int coefs_precision;

    // use the same buffer for normalized coefficients
    kk = (INT16 *) prekk;
    coefs_precision = normalize_coeffs_8bpc(xout, ksize, prekk);

    yy = 0;
    for (; yy < yout - 3; yy += 4) {
        ImagingResampleHorizontalConvolution8u4x(
            unpacked_output_p + yy * xout,
            unpacked_output_p + (yy + 1) * xout,
            unpacked_output_p + (yy + 2) * xout,
            unpacked_output_p + (yy + 3) * xout,
            unpacked_input_p + (yy + offset) * xin,
            unpacked_input_p + (yy + offset + 1) * xin,
            unpacked_input_p + (yy + offset + 2) * xin,
            unpacked_input_p + (yy + offset + 3) * xin,
            xout, bounds, kk, ksize,
            coefs_precision
        );
    }
    for (; yy < yout; yy++) {
        ImagingResampleHorizontalConvolution8u(
            unpacked_output_p + yy * xout,
            unpacked_input_p + (yy + offset) * xin,
            xout, bounds, kk, ksize,
            coefs_precision
        );
    }
}

void ImagingResampleVertical_8bpc(
    UINT32 * unpacked_output_p , UINT32 * unpacked_input_p, int offset, int ksize, int *bounds, double *prekk, int xout, int yout) {
    int ss0;
    int xx, yy, y, ymin, ymax;
    INT16 *k, *kk;
    int coefs_precision;

    // use the same buffer for normalized coefficients
    kk = (INT16 *) prekk;
    coefs_precision = normalize_coeffs_8bpc(yout, ksize, prekk);

    for (yy = 0; yy < yout; yy++) {
        k = &kk[yy * ksize];
        ymin = bounds[yy * 2 + 0];
        ymax = bounds[yy * 2 + 1];
        ImagingResampleVerticalConvolution8u(unpacked_output_p + yy * xout, unpacked_input_p, ymin, ymax, k, coefs_precision, xout);
    }
}

UINT32 * ImagingResampleInner(UINT32 * unpacked_input_p, int xin, int yin, int xout, int yout) {
    int i, need_horizontal, need_vertical;
    int ybox_first, ybox_last;
    int ksize_horiz, ksize_vert;
    int *bounds_horiz, *bounds_vert;
    double *kk_horiz, *kk_vert;
    UINT32 * unpacked_output_p = NULL;
    UINT32 * unpacked_output_temp_p = NULL;

    need_horizontal = xout != xin;
    need_vertical = yout != yin;

    ksize_horiz = precompute_coeffs(xin, xout, &bounds_horiz, &kk_horiz);
    // if (!ksize_horiz) {
    //     return NULL;
    // }

    ksize_vert = precompute_coeffs(yin, yout, &bounds_vert, &kk_vert);
    // if (!ksize_vert) {
    //     free(bounds_horiz);
    //     free(kk_horiz);
    //     return NULL;
    // }

    // First used row in the source image
    ybox_first = bounds_vert[0];
    // Last used row in the source image
    ybox_last = bounds_vert[yout * 2 - 2] + bounds_vert[yout * 2 - 1];

    /* two-pass resize, horizontal pass */
    if (need_horizontal) {
        // Shift bounds for vertical pass
        for (i = 0; i < yout; i++) {
            bounds_vert[i * 2] -= ybox_first;
        }

        // imTemp = ImagingNewDirty(imIn->mode, xout, ybox_last - ybox_first);
        // if (imTemp) {
        //     ResampleHorizontal(
        //         imTemp, imIn, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
        // }
        unpacked_output_temp_p = (UINT32 *) malloc(sizeof(UINT32) * xout * yin);
        ImagingResampleHorizontal_8bpc(unpacked_output_temp_p, unpacked_input_p, ybox_first, ksize_horiz, bounds_horiz, kk_horiz, xout, yin, xin);
        free(bounds_horiz);
        free(kk_horiz);
        // if (!imTemp) {
        //     free(bounds_vert);
        //     free(kk_vert);
        //     return NULL;
        // }
        // imOut = imIn = imTemp;
        unpacked_output_p = unpacked_input_p = unpacked_output_temp_p;
    } else {
        // Free in any case
        free(bounds_horiz);
        free(kk_horiz);
    }

    /* vertical pass */
    if (need_vertical) {
        // imOut = ImagingNewDirty(imIn->mode, imIn->xsize, ysize);
        unpacked_output_p = (UINT32 *) malloc(sizeof(UINT32) * xout * yout);
        // if (imOut) {
            ImagingResampleVertical_8bpc(unpacked_output_p, unpacked_input_p, 0, ksize_vert, bounds_vert, kk_vert, xout, yout);
        // }
        // /* it's safe to call ImagingDelete with empty value
        //    if previous step was not performed. */
        // ImagingDelete(imTemp);
        free(unpacked_output_temp_p);
        free(bounds_vert);
        free(kk_vert);
        // if (!imOut) {
        //     return NULL;
        // }
    } else {
        // Free in any case
        free(bounds_vert);
        free(kk_vert);
    }

    // /* none of the previous steps are performed, copying */
    // if (!imOut) {
    //     imOut = ImagingCopy(imIn);
    // }

    return unpacked_output_p;
}

void beepidiboop(const Tensor& input, const Tensor& output) {
  // Assume shape is 1, 3, H, W  and layout is channels_last

   auto xin = input.size(3);
   auto yin = input.size(2);
   auto xout = output.size(3);
   auto yout = output.size(2);
   auto num_pixels_input = xin * yin;
   auto num_pixels_output = xout * yout;
  

  UINT32 * unpacked_input_p = (UINT32 *) malloc(sizeof(UINT32) * num_pixels_input);
  const uint8_t * packed_input_p = (const uint8_t *) input.data_ptr<uint8_t>();
  unpack_rgb((uint8_t *)unpacked_input_p, packed_input_p, num_pixels_input);

  UINT32 * unpacked_output_p = ImagingResampleInner(unpacked_input_p, xin, yin, xout, yout);

  uint8_t * packed_output_p = (uint8_t *) output.data_ptr<uint8_t>();
  pack_rgb(packed_output_p, (const uint8_t *)unpacked_output_p, num_pixels_output);
}


} // anonymous namespace
} // namespace native
} // namespace at
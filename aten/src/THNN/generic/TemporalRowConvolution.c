#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/TemporalRowConvolution.c"
#else

static inline void THNN_(TemporalRowConvolution_shapeCheck)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *weight,
        THTensor *bias,
        int kW,
        int dW,
        int padW) {

        THArgCheck(kW > 0, 5,
                   "kernel size should be greater than zero, but got kW: %d", kW);
        THArgCheck(dW > 0, 6,
                   "stride should be greater than zero, but got dW: %d", dW);
        THNN_ARGCHECK(!weight->is_empty() && weight->dim() == 3, 3, weight,
                      "non-empty 3D weight tensor expected, but got: %s");
    THArgCheck(THTensor_(isContiguous)(weight), 4, "weight must be contiguous");
    THArgCheck(!bias || THTensor_(isContiguous)(bias), 5, "bias must be contiguous");

        if (bias != NULL) {
                THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size(0));
        }

        // we're always looking at (possibly batch) x feats x seq
        int ndim = input->dim();
        int dimF = 0;
        int dimS = 1;

        if (ndim == 3) {
                ++dimS;
                ++dimF;
        }

        THNN_ARGCHECK(!input->is_empty() && (ndim == 2 || ndim == 3), 1, input,
                      "non-empty 2D or 3D (batch mode) input tensor expected, but got :%s");

        int64_t inputFrameSize = THTensor_sizeLegacyNoScalars(weight, 0);
        int64_t nInputFrame = input->size(dimS);
        int64_t nOutputFrame = (nInputFrame + 2 * padW - kW) / dW + 1;

        if (nOutputFrame < 1) {
                THError("Given input size: (%d x %d). "
                        "Calculated output size: (%d x %d). Output size is too small",
                        inputFrameSize, nInputFrame, inputFrameSize, nOutputFrame);
        }

        THNN_CHECK_DIM_SIZE(input, ndim, dimF, inputFrameSize);

        if (gradOutput != NULL) {
                THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimF, inputFrameSize);
                THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimS, nOutputFrame);
        }
}

static void THNN_(unfolded_acc_row)(
        THTensor *finput,
        THTensor *input,
        int kW,
        int dW,
        int padW,
        int64_t inputFrameSize,
        int64_t nInputFrame,
        int64_t nOutputFrame) {

        int64_t c;
        scalar_t *input_data = input->data<scalar_t>();
        scalar_t *finput_data = finput->data<scalar_t>();

// #pragma omp parallel for private(c)
        for (c = 0; c < inputFrameSize; c++) {
                int64_t kw, x;
                int64_t ix = 0;

                for (kw = 0; kw < kW; kw++) {
                        scalar_t *src = finput_data
                                    + c * (kW * nOutputFrame)
                                    + kw * (nOutputFrame);
                        scalar_t *dst = input_data + c * (nInputFrame);

                        ix = (size_t)(kw);
                        if (dW == 1) {
                          scalar_t *dst_slice = dst + (size_t)(ix);
                          THVector_(cadd)(dst_slice, dst_slice, src, 1, nOutputFrame);
                        } else {
                                for (x = 0; x < nOutputFrame; x++) {
                                  scalar_t *dst_slice = dst + (size_t)(ix + x * dW);
                                  THVector_(cadd)(dst_slice, dst_slice,
                                                  src + (size_t)(x), 1, 1);
                                }
                        }
                }
        }
}

static void THNN_(unfolded_copy_row)(
        THTensor *finput,
        THTensor *input,
        int kW,
        int dW,
        int padW,
        int64_t inputFrameSize,
        int64_t nInputFrame,
        int64_t nOutputFrame) {

        int64_t k;
        scalar_t *input_data = input->data<scalar_t>();
        scalar_t *finput_data = finput->data<scalar_t>();

// #pragma omp parallel for private(k)
        for (k = 0; k < inputFrameSize * kW; k++) {
                int64_t c = k / kW;
                int64_t rest = k % kW;
                int64_t kw = rest % kW;
                int64_t x;
                int64_t ix;
                scalar_t *dst = finput_data + c * (kW * nOutputFrame) + kw * (nOutputFrame);
                scalar_t *src = input_data + c * (nInputFrame);

                ix = (size_t)(kw);
                if (dW == 1) {
                        memcpy(dst, src+(size_t)(ix), sizeof(scalar_t) * (nOutputFrame));
                } else {
                        for (x = 0; x < nOutputFrame; x++) {
                                memcpy(dst + (size_t)(x), src + (size_t)(ix + x * dW),
                                       sizeof(scalar_t) * 1);
                        }
                }
        }
}

static void THNN_(TemporalRowConvolution_updateOutput_frame)(
        THTensor *input,
        THTensor *output,
        THTensor *weight,
        THTensor *bias,
        THTensor *finput,
        int kW,
        int dW,
        int padW,
        int64_t inputFrameSize,
        int64_t nInputFrame,
        int64_t nOutputFrame) {

        int64_t i;

        THTensor *output3d = THTensor_(newWithStorage3d)(
                THTensor_getStoragePtr(output), output->storage_offset(),
                inputFrameSize, -1,
                1, -1,
                nOutputFrame, -1);

        THNN_(unfolded_copy_row)(finput, input, kW, dW, padW,
                                 inputFrameSize, nInputFrame, nOutputFrame);

        THTensor_(zero)(output);

        if (bias != NULL) {
                for (i = 0; i < inputFrameSize; i++)
                        THVector_(fill)
                                (THStorage_(data)(THTensor_getStoragePtr(output)) + output->storage_offset()
                                + output->stride(0) * i,
                                THTensor_(get1d)(bias, i), nOutputFrame);
        }

        THTensor_(baddbmm)(output3d, 1, output3d, 1, weight, finput);

        c10::raw::intrusive_ptr::decref(output3d);
}

void THNN_(TemporalRowConvolution_updateOutput)(
        THNNState *state,
        THTensor *input,
        THTensor *output,
        THTensor *weight,
        THTensor *bias,
        THTensor *finput,
        THTensor *fgradInput,     // unused here but needed for Cuda
        int kW,
        int dW,
        int padW,
        bool featFirst) {

        int ndim = input->dim();

        THTensor *tinput = NULL;
        if (!featFirst) {
                tinput = THTensor_(newTranspose)(input, ndim - 1, ndim - 2);
                input = THTensor_(newContiguous)(tinput);
        } else {
                input = THTensor_(newContiguous)(input);
        }

        THNN_(TemporalRowConvolution_shapeCheck)(
                state, input, NULL, weight, bias, kW, dW, padW);

        int64_t inputFrameSize = THTensor_sizeLegacyNoScalars(weight, 0);
        int64_t nInputFrame = input->size(ndim - 1);
        int64_t nOutputFrame = (nInputFrame + 2 * padW - kW) / dW + 1;

        if (ndim == 2) { /* non-batch mode */

                THTensor_(resize3d)(finput, inputFrameSize, kW, nOutputFrame);
                THTensor_(resize2d)(output, inputFrameSize, nOutputFrame);

                THTensor_(zero)(finput);
                THTensor_(zero)(output);

                THNN_(TemporalRowConvolution_updateOutput_frame)
                        (input, output, weight, bias, finput,
                        kW, dW, padW,
                        inputFrameSize, nInputFrame, nOutputFrame);

        } else {
                int64_t T = input->size(0);
                int64_t t;

                THTensor_(resize4d)(finput, T, inputFrameSize, kW, nOutputFrame);
                THTensor_(resize3d)(output, T, inputFrameSize, nOutputFrame);

                THTensor_(zero)(finput);
                THTensor_(zero)(output);

#pragma omp parallel for private(t)
                for (t = 0; t < T; t++) {
                        THTensor *input_t = THTensor_(newSelect)(input, 0, t);
                        THTensor *output_t = THTensor_(newSelect)(output, 0, t);
                        THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

                        THNN_(TemporalRowConvolution_updateOutput_frame)
                                (input_t, output_t, weight, bias, finput_t,
                                kW, dW, padW, inputFrameSize, nInputFrame, nOutputFrame);

                        c10::raw::intrusive_ptr::decref(input_t);
                        c10::raw::intrusive_ptr::decref(output_t);
                        c10::raw::intrusive_ptr::decref(finput_t);
                }
        }

        if (!featFirst) { // NOTE: output will NOT be contiguous in this case
                THTensor_(transpose)(output, output, ndim - 1, ndim - 2);
                c10::raw::intrusive_ptr::decref(tinput);
        }

        c10::raw::intrusive_ptr::decref(input);
}

static void THNN_(TemporalRowConvolution_updateGradInput_frame)(
        THTensor *gradInput,
        THTensor *gradOutput,
        THTensor *weight,
        THTensor *fgradInput,
        int kW,
        int dW,
        int padW,
        int64_t inputFrameSize,
        int64_t nInputFrame,
        int64_t nOutputFrame) {

        THTensor *gradOutput3d = THTensor_(newWithStorage3d)(
                THTensor_getStoragePtr(gradOutput), gradOutput->storage_offset(),
                inputFrameSize, -1,
                1, -1,
                nOutputFrame, -1);

        // weight:                        inputFrameSize x kW x 1
        // gradOutput3d:        inputFrameSize x 1 x nOutputFrame
        THTensor_(baddbmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput3d);
        // fgradInput:                inputFrameSize x kW x nOutputFrame
        c10::raw::intrusive_ptr::decref(gradOutput3d);

        THTensor_(zero)(gradInput);

        THNN_(unfolded_acc_row)(fgradInput, gradInput,
                                kW, dW, padW,
                                inputFrameSize, nInputFrame, nOutputFrame);
}

void THNN_(TemporalRowConvolution_updateGradInput)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *gradInput,
        THTensor *weight,
        THTensor *finput,
        THTensor *fgradInput,
        int kW,
        int dW,
        int padW,
        bool featFirst) {

        int ndim = input->dim();

        THTensor *tinput, *tgradOutput;

        if (!featFirst) {
                tinput = THTensor_(newTranspose)(input, ndim - 1, ndim - 2);
                tgradOutput = THTensor_(newTranspose)(gradOutput, ndim - 1, ndim - 2);

                input = THTensor_(newContiguous)(tinput);
                gradOutput = THTensor_(newContiguous)(tgradOutput);

        } else {
                input = THTensor_(newContiguous)(input);
                gradOutput = THTensor_(newContiguous)(gradOutput);
        }

        THNN_(TemporalRowConvolution_shapeCheck)(state, input, gradOutput, weight,
                                                 NULL, kW, dW, padW);

        int64_t inputFrameSize = THTensor_sizeLegacyNoScalars(weight, 0);
        int64_t nInputFrame = input->size(ndim - 1);
        int64_t nOutputFrame = (nInputFrame + 2 * padW - kW) / dW + 1;

        THTensor_(resizeAs)(fgradInput, finput);
        THTensor_(resizeAs)(gradInput, input);

        THTensor_(zero)(fgradInput);
        THTensor_(zero)(gradInput);

    THTensor *tweight = THTensor_(new)();
    THTensor_(transpose)(tweight, weight, 1, 2);

        if (ndim == 2) {
                THNN_(TemporalRowConvolution_updateGradInput_frame)
                        (gradInput, gradOutput, tweight, fgradInput,
                        kW, dW, padW,
                        inputFrameSize, nInputFrame, nOutputFrame);
        } else {
                int64_t T = input->size(0);
                int64_t t;

#pragma omp parallel for private(t)
                for (t = 0; t < T; t++) {

                        THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
                        THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
                        THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

                        THNN_(TemporalRowConvolution_updateGradInput_frame)
                                (gradInput_t, gradOutput_t, tweight, fgradInput_t,
                                kW, dW, padW,
                                inputFrameSize, nInputFrame, nOutputFrame);

                        c10::raw::intrusive_ptr::decref(gradInput_t);
                        c10::raw::intrusive_ptr::decref(gradOutput_t);
                        c10::raw::intrusive_ptr::decref(fgradInput_t);
                }
        }

    c10::raw::intrusive_ptr::decref(tweight);

        if (!featFirst) { // NOTE: gradInput will NOT be contiguous in this case

                c10::raw::intrusive_ptr::decref(tinput);
                c10::raw::intrusive_ptr::decref(tgradOutput);

                THTensor_(transpose)(gradInput, gradInput, ndim - 1, ndim - 2);
        }

        c10::raw::intrusive_ptr::decref(input);
        c10::raw::intrusive_ptr::decref(gradOutput);

}

static void THNN_(TemporalRowConvolution_accGradParameters_frame)(
        THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias,
        THTensor *finput, scalar_t scale) {

        int64_t i;
        THTensor *gradOutput3d = THTensor_(newWithStorage3d)(
                THTensor_getStoragePtr(gradOutput), gradOutput->storage_offset(),
                gradOutput->size(0), -1,
                1, -1,
                gradOutput->size(1), -1);

    THTensor *tfinput = THTensor_(new)();
        THTensor_(transpose)(tfinput, finput, 1, 2);
        // gradOutput3d:        inputFrameSize x 1 x nOutputFrame
        // finput:                        inputFrameSize x nOutputFrame x kW
        THTensor_(baddbmm)(gradWeight, 1, gradWeight, scale, gradOutput3d, tfinput);
        // gradWeight:                inputFrameSize x 1 x kW
    c10::raw::intrusive_ptr::decref(tfinput);

        if (gradBias != NULL) {
                for (i = 0; i < THTensor_sizeLegacyNoScalars(gradBias, 0); i++) {
                        int64_t k;
                        scalar_t sum = 0;
                        scalar_t *data = THStorage_(data)(THTensor_getStoragePtr(gradOutput3d))
                                     + gradOutput3d->storage_offset()
                                     + i * gradOutput3d->stride(0);
                        for (k = 0; k < gradOutput3d->size(2); k++) {
                                sum += data[k];
                        }
                        (THStorage_(data)(THTensor_getStoragePtr(gradBias)) + gradBias->storage_offset())[i]
                                += scale * sum;
                }
        }

        c10::raw::intrusive_ptr::decref(gradOutput3d);

}

void THNN_(TemporalRowConvolution_accGradParameters)(
        THNNState *state,
        THTensor *input,
        THTensor *gradOutput,
        THTensor *gradWeight,
        THTensor *gradBias,
        THTensor *finput,
        THTensor *fgradInput,
        int kW,
        int dW,
        int padW,
        bool featFirst,
        accreal scale_) {

    scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
        int ndim = input->dim();

        THTensor *tinput = NULL;
        THTensor *tgradOutput = NULL;

        if (!featFirst) {
                tinput = THTensor_(newTranspose)(input, ndim - 1, ndim - 2);
                tgradOutput = THTensor_(newTranspose)(gradOutput, ndim - 1, ndim - 2);

                input = THTensor_(newContiguous)(tinput);
                gradOutput = THTensor_(newContiguous)(tgradOutput);
        } else {
                input = THTensor_(newContiguous)(input);
                gradOutput = THTensor_(newContiguous)(gradOutput);
        }

        THNN_(TemporalRowConvolution_shapeCheck)
                (state, input, gradOutput, gradWeight, gradBias, kW, dW, padW);

        if (ndim == 2) {
                THNN_(TemporalRowConvolution_accGradParameters_frame)(
                        gradOutput, gradWeight, gradBias, finput, scale);
        } else {
                int64_t T = input->size(0);
                int64_t t;

                for (t = 0; t < T; t++) {
                        THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
                        THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

                        THNN_(TemporalRowConvolution_accGradParameters_frame)(
                                gradOutput_t, gradWeight, gradBias, finput_t, scale);

                        c10::raw::intrusive_ptr::decref(gradOutput_t);
                        c10::raw::intrusive_ptr::decref(finput_t);
                }
        }

        if (!featFirst) {
                c10::raw::intrusive_ptr::decref(tinput);
                c10::raw::intrusive_ptr::decref(tgradOutput);
        }

        c10::raw::intrusive_ptr::decref(input);
        c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif

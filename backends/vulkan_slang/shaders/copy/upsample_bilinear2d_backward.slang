// Upsample bilinear 2D backward: accumulate grad_output into grad_input.
// Thread per INPUT pixel. Each input pixel gathers weighted gradients from
// all output pixels whose bilinear interpolation touches it.

struct Params {
    uint numel;          // total input elements (N*C*in_h*in_w)
    uint batch_channels; // N * C
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint align_corners;  // 1 = align_corners, 0 = half-pixel offset
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;   // [N, C, out_h, out_w]
[[vk::binding(1)]] RWStructuredBuffer<float> grad_input;   // [N, C, in_h, in_w]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    uint in_spatial = params.in_h * params.in_w;
    uint bc = tid.x / in_spatial;
    uint spatial_idx = tid.x % in_spatial;
    uint ih = spatial_idx / params.in_w;
    uint iw = spatial_idx % params.in_w;

    // For each output pixel (oh, ow), the forward pass computes:
    //   src_h = f(oh), src_w = g(ow)
    //   h0 = floor(src_h), h1 = h0+1, fh = src_h - h0
    //   w0 = floor(src_w), w1 = w0+1, fw = src_w - w0
    //   out = input[h0,w0]*(1-fh)*(1-fw) + input[h0,w1]*(1-fh)*fw
    //       + input[h1,w0]*fh*(1-fw)     + input[h1,w1]*fh*fw
    //
    // So grad_input[ih,iw] += grad_output[oh,ow] * weight
    // where weight depends on whether (ih,iw) is the (h0,w0), (h0,w1), (h1,w0), or (h1,w1) corner.
    //
    // We need to find the range of output pixels that could reference input pixel (ih, iw).
    // From the coordinate mapping, we can bound the output pixel range.

    float rh, rw;
    if (params.align_corners != 0 && params.out_h > 1) {
        rh = float(params.out_h - 1) / float(params.in_h - 1);
    } else {
        rh = float(params.out_h) / float(params.in_h);
    }
    if (params.align_corners != 0 && params.out_w > 1) {
        rw = float(params.out_w - 1) / float(params.in_w - 1);
    } else {
        rw = float(params.out_w) / float(params.in_w);
    }

    // An output pixel at (oh,ow) has src_h that maps to input range [h0, h1].
    // Input pixel ih is touched when h0==ih or h1==ih, i.e., src_h in [ih-1, ih+1].
    // We need to find oh range where src_h falls in [ih-1, ih+1].

    // Compute output pixel range that could reference this input pixel
    uint oh_start, oh_end, ow_start, ow_end;

    if (params.align_corners != 0 && params.out_h > 1) {
        // src_h = oh * (in_h-1) / (out_h-1)
        // oh = src_h * (out_h-1) / (in_h-1)
        // src_h in [ih-1, ih+1] => oh in [max(0, (ih-1)*rh), min(out_h-1, (ih+1)*rh)]
        float oh_min_f = float(ih > 0 ? ih - 1 : 0) * rh;
        float oh_max_f = float(min(ih + 1, params.in_h - 1)) * rh;
        oh_start = uint(oh_min_f);
        oh_end = min(uint(oh_max_f) + 1, params.out_h);
    } else {
        // src_h = (oh + 0.5) * in_h / out_h - 0.5
        // oh = (src_h + 0.5) * out_h / in_h - 0.5
        float src_h_min = max(float(ih) - 1.0f, -0.5f);
        float src_h_max = min(float(ih) + 1.0f, float(params.in_h - 1) + 0.5f);
        float oh_min_f = (src_h_min + 0.5f) * rh - 0.5f;
        float oh_max_f = (src_h_max + 0.5f) * rh - 0.5f;
        oh_start = oh_min_f > 0.0f ? uint(oh_min_f) : 0;
        oh_end = min(uint(oh_max_f) + 1, params.out_h);
    }

    if (params.align_corners != 0 && params.out_w > 1) {
        float ow_min_f = float(iw > 0 ? iw - 1 : 0) * rw;
        float ow_max_f = float(min(iw + 1, params.in_w - 1)) * rw;
        ow_start = uint(ow_min_f);
        ow_end = min(uint(ow_max_f) + 1, params.out_w);
    } else {
        float src_w_min = max(float(iw) - 1.0f, -0.5f);
        float src_w_max = min(float(iw) + 1.0f, float(params.in_w - 1) + 0.5f);
        float ow_min_f = (src_w_min + 0.5f) * rw - 0.5f;
        float ow_max_f = (src_w_max + 0.5f) * rw - 0.5f;
        ow_start = ow_min_f > 0.0f ? uint(ow_min_f) : 0;
        ow_end = min(uint(ow_max_f) + 1, params.out_w);
    }

    uint out_spatial = params.out_h * params.out_w;
    uint out_base = bc * out_spatial;

    float sum = 0.0f;
    for (uint oh = oh_start; oh < oh_end; oh++) {
        // Compute src_h for this output pixel
        float src_h;
        if (params.align_corners != 0 && params.out_h > 1) {
            src_h = float(oh) * float(params.in_h - 1) / float(params.out_h - 1);
        } else {
            src_h = (float(oh) + 0.5f) * float(params.in_h) / float(params.out_h) - 0.5f;
        }
        src_h = max(0.0f, min(src_h, float(params.in_h - 1)));

        uint h0 = uint(src_h);
        uint h1 = min(h0 + 1, params.in_h - 1);
        float fh = src_h - float(h0);

        // Check if this input pixel is h0 or h1 for this output pixel
        float wh;
        if (ih == h0 && ih == h1) {
            wh = 1.0f;  // h0 == h1 (edge case: in_h == 1 or at boundary)
        } else if (ih == h0) {
            wh = 1.0f - fh;
        } else if (ih == h1) {
            wh = fh;
        } else {
            continue;  // This output pixel doesn't reference our input row
        }

        for (uint ow = ow_start; ow < ow_end; ow++) {
            float src_w;
            if (params.align_corners != 0 && params.out_w > 1) {
                src_w = float(ow) * float(params.in_w - 1) / float(params.out_w - 1);
            } else {
                src_w = (float(ow) + 0.5f) * float(params.in_w) / float(params.out_w) - 0.5f;
            }
            src_w = max(0.0f, min(src_w, float(params.in_w - 1)));

            uint w0 = uint(src_w);
            uint w1 = min(w0 + 1, params.in_w - 1);
            float fw = src_w - float(w0);

            float ww;
            if (iw == w0 && iw == w1) {
                ww = 1.0f;
            } else if (iw == w0) {
                ww = 1.0f - fw;
            } else if (iw == w1) {
                ww = fw;
            } else {
                continue;
            }

            sum += grad_output[out_base + oh * params.out_w + ow] * wh * ww;
        }
    }

    grad_input[tid.x] = sum;
}

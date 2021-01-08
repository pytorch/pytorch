#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/cpu/ChannelShuffleKernel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void cpu_channel_shuffle(
    Tensor& output,
    const Tensor& input,
    int64_t groups) {
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t channels_per_group = channels / groups;
  int64_t image_size = input.numel() / nbatch / channels;

  // treat input tensor as shape of [n, g, oc, ...]
  // output tensor as shape of [n, oc, g, ...]
  //
  // parallel on dimension of n, oc, g
  using Vec = vec256::Vec256<scalar_t>;
  at::parallel_for (0, nbatch * /* oc*g */channels, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oc = 0;
    int64_t g = 0;
    data_index_init(begin, n, nbatch, oc, channels_per_group, g, groups);

    for (int64_t i = begin; i < end; i++) {
      scalar_t* output_ptr = output_data + i * image_size;
      scalar_t* input_ptr = input_data + n * channels * image_size +
          g * channels_per_group * image_size + oc * image_size;
      vec256::map(
         [](Vec x) { return x; },
         output_ptr,
         input_ptr,
         image_size);

      // move on to next output index
      data_index_step(n, nbatch, oc, channels_per_group, g, groups);
    }
  });
}

// abstract parallel routine so as to avoid condition checks on perf critical paths.
template <typename scalar_t, typename func_t>
static void parallel_nd(scalar_t* input_data, scalar_t* output_data,
    int64_t M, int64_t C, const func_t& f) {
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      scalar_t* output_ptr = output_data + i * C;
      scalar_t* input_ptr = input_data + i * C;
      f(input_ptr, output_ptr);
    }
  });
}

template <typename scalar_t>
void cpu_channel_shuffle_cl(
    Tensor& output,
    const Tensor& input,
    int64_t groups) {
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t M = input.numel() / channels;
  int64_t N = channels / groups;
  int64_t G = groups;

  bool is_float32 = input.scalar_type() == kFloat;
  // treat input tensor as shape of [M, G, N]
  // treat output tensor as shape of [M, N, G]
  //
  // add vectorized path for float32 when groups = 2, 4, 8, 16, ...
  // to assure similar perf as NCHW memory format.
  using Vec = vec256::Vec256<scalar_t>;
  int64_t inner_size = N - (N % Vec::size());
  if (G == 2 && is_float32) {
    parallel_nd(input_data, output_data, M, channels, [&](scalar_t* in, scalar_t* out) {
      int64_t k = 0;
      for (; k < inner_size; k += Vec::size()) {
        // a = {a0, a1, a2, a3, a4, a5, a6, a7}
        // b = {b0, b1, b2, b3, b4, b5, b6, b7}
        Vec a = Vec::loadu(in + k);
        Vec b = Vec::loadu(in + N + k);

        // ab0 = {a0, b0, a1, b1, a2, b2, a3, b3}
        // ab1 = {a4, b4, a5, b5, a6, b6, a7, b7}
        Vec ab0, ab1;
        std::tie(ab0, ab1) = vec256::interleave2(a, b);

        ab0.store(out + k * 2);
        ab1.store(out + k * 2 + Vec::size());
      }
      for (; k < N; k++) {
        out[k * 2] = in[k];
        out[k * 2 + 1] = in[k + N];
      }
    });
  } else if (G == 4 && is_float32) {
    parallel_nd(input_data, output_data, M, channels, [&](scalar_t* in, scalar_t* out) {
      int64_t k = 0;
      for (; k < inner_size; k += Vec::size()) {
        // a = {a0, a1, a2, a3, a4, a5, a6, a7}
        // b = {b0, b1, b2, b3, b4, b5, b6, b7}
        // c = {c0, c1, c2, c3, c4, c5, c6, c7}
        // d = {d0, d1, d2, d3, d4, d5, d6, d7}
        Vec a = Vec::loadu(in + k);
        Vec b = Vec::loadu(in + N + k);
        Vec c = Vec::loadu(in + 2 * N + k);
        Vec d = Vec::loadu(in + 3 * N + k);

        // ac0 = {a0, c0, a1, c1, a2, c2, a3, c3}
        // ac1 = {a4, c4, a5, c5, a6, c6, a7, c7}
        // bd0 = {b0, d0, b1, d1, b2, d2, b3, d3}
        // bd1 = {b4, d4, b5, d5, b6, d6, b7, d7}
        Vec ac0, ac1, bd0, bd1;
        std::tie(ac0, ac1) = vec256::interleave2(a, c);
        std::tie(bd0, bd1) = vec256::interleave2(b, d);

        // abcd0 = {a0, b0, c0, d0, a1, b1, c1, d1}
        // abcd1 = {a2, b2, c2, d2, a3, b3, c3, d3}
        // abcd2 = {a4, b4, c4, d4, a5, b5, c5, d5}
        // abcd3 = {a6, b6, c6, d6, a7, b7, c7, d7}
        Vec abcd0, abcd1, abcd2, abcd3;
        std::tie(abcd0, abcd1) = vec256::interleave2(ac0, bd0);
        std::tie(abcd2, abcd3) = vec256::interleave2(ac1, bd1);

        abcd0.store(out + k * 4);
        abcd1.store(out + k * 4 + Vec::size());
        abcd2.store(out + k * 4 + Vec::size() * 2);
        abcd3.store(out + k * 4 + Vec::size() * 3);
      }
      for (; k < N; k++) {
        out[k * 4] = in[k];
        out[k * 4 + 1] = in[k + N];
        out[k * 4 + 2] = in[k + 2 * N];
        out[k * 4 + 3] = in[k + 3 * N];
      }
    });
  } else if (G % Vec::size() == 0 && is_float32) { // groups = 8, 16, ...
    parallel_nd(input_data, output_data, M, channels, [&](scalar_t* in, scalar_t* out) {
      for (int64_t grp = 0 ; grp < G; grp += Vec::size()) {
        int64_t k = 0;
        for (; k < inner_size; k += Vec::size()) {
          // a = {a0, a1, a2, a3, a4, a5, a6, a7}
          // b = {b0, b1, b2, b3, b4, b5, b6, b7}
          // c = {c0, c1, c2, c3, c4, c5, c6, c7}
          // d = {d0, d1, d2, d3, d4, d5, d6, d7}
          Vec a = Vec::loadu(in + (grp + 0) * N + k);
          Vec b = Vec::loadu(in + (grp + 1) * N + k);
          Vec c = Vec::loadu(in + (grp + 2) * N + k);
          Vec d = Vec::loadu(in + (grp + 3) * N + k);

          // e = {e0, e1, e2, e3, e4, e5, e6, e7}
          // f = {f0, f1, f2, f3, f4, f5, f6, f7}
          // g = {g0, g1, g2, g3, g4, g5, g6, g7}
          // h = {h0, h1, h2, h3, h4, h5, h6, h7}
          Vec e = Vec::loadu(in + (grp + 4) * N + k);
          Vec f = Vec::loadu(in + (grp + 5) * N + k);
          Vec g = Vec::loadu(in + (grp + 6) * N + k);
          Vec h = Vec::loadu(in + (grp + 7) * N + k);

          // ae0 = {a0, e0, a1, e1, a2, e2, a3, e3}
          // ae1 = {a4, e4, a5, e5, a6, e6, a7, e7}
          // bf0 = {b0, f0, b1, f1, b2, f2, b3, f3}
          // bf1 = {b4, f4, b5, f5, b6, f6, b7, f7}
          Vec ae0, ae1, bf0, bf1;
          std::tie(ae0, ae1) = vec256::interleave2(a, e);
          std::tie(bf0, bf1) = vec256::interleave2(b, f);

          // cg0 = {c0, g0, c1, g1, c2, g2, c3, g3}
          // cg1 = {c4, g4, c5, g5, c6, g6, c7, g7}
          // dh0 = {d0, h0, d1, h1, d2, h2, d3, h3}
          // dh1 = {d4, h4, d5, h5, d6, h6, d7, h7}
          Vec cg0, cg1, dh0, dh1;
          std::tie(cg0, cg1) = vec256::interleave2(c, g);
          std::tie(dh0, dh1) = vec256::interleave2(d, h);

          // aceg0 = {a0, c0, e0, g0, a1, c1, e1, g1}
          // aceg1 = {a2, c2, e2, g2, a3, c3, e3, g3}
          // aceg2 = {a4, c4, e4, g4, a5, c5, e5, g5}
          // aceg3 = {a6, c6, e6, g6, a7, c7, e7, g7}
          Vec aceg0, aceg1, aceg2, aceg3;
          std::tie(aceg0, aceg1) = vec256::interleave2(ae0, cg0);
          std::tie(aceg2, aceg3) = vec256::interleave2(ae1, cg1);

          // bdfh0 = {b0, d0, f0, h0, b1, d1, f1, h1}
          // bdfh1 = {b2, d2, f2, h2, b3, d3, f3, h3}
          // bdfh2 = {b4, d4, f4, h4, b5, d5, f5, h5}
          // bdfh3 = {b6, d6, f6, h6, b7, d7, f7, h7}
          Vec bdfh0, bdfh1, bdfh2, bdfh3;
          std::tie(bdfh0, bdfh1) = vec256::interleave2(bf0, dh0);
          std::tie(bdfh2, bdfh3) = vec256::interleave2(bf1, dh1);

          // y_i = {a_i, b_i, c_i, d_i, e_i, f_i, g_i, h_i} : where i = 0:7
          Vec y0, y1, y2, y3, y4, y5, y6, y7;
          std::tie(y0, y1) = vec256::interleave2(aceg0, bdfh0);
          std::tie(y2, y3) = vec256::interleave2(aceg1, bdfh1);
          std::tie(y4, y5) = vec256::interleave2(aceg2, bdfh2);
          std::tie(y6, y7) = vec256::interleave2(aceg3, bdfh3);

          y0.store(out + (k + 0) * G + grp);
          y1.store(out + (k + 1) * G + grp);
          y2.store(out + (k + 2) * G + grp);
          y3.store(out + (k + 3) * G + grp);
          y4.store(out + (k + 4) * G + grp);
          y5.store(out + (k + 5) * G + grp);
          y6.store(out + (k + 6) * G + grp);
          y7.store(out + (k + 7) * G + grp);
        }
        for (; k < N; k++) {
          out[k * G + (grp + 0)] = in[(grp + 0) * N + k];
          out[k * G + (grp + 1)] = in[(grp + 1) * N + k];
          out[k * G + (grp + 2)] = in[(grp + 2) * N + k];
          out[k * G + (grp + 3)] = in[(grp + 3) * N + k];
          out[k * G + (grp + 4)] = in[(grp + 4) * N + k];
          out[k * G + (grp + 5)] = in[(grp + 5) * N + k];
          out[k * G + (grp + 6)] = in[(grp + 6) * N + k];
          out[k * G + (grp + 7)] = in[(grp + 7) * N + k];
        }
      }
    });
  } else {
    // float32 when groups != 2, 4, 8, 16, ... and float64
    parallel_nd(input_data, output_data, M, channels, [&](scalar_t* in, scalar_t* out) {
      for (int64_t oc = 0; oc < N; oc++) {
        for (int64_t g = 0; g < G; g++) {
          out[oc * G + g] = in[g * N + oc];
        }
      }
    });
  }
}

void channel_shuffle_kernel_impl(
    Tensor& output,
    const Tensor& input,
    int64_t groups) {
  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_shuffle", [&] {
        cpu_channel_shuffle<scalar_t>(output, input, groups);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_shuffle_cl", [&] {
        cpu_channel_shuffle_cl<scalar_t>(output, input, groups);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(channel_shuffle_kernel, &channel_shuffle_kernel_impl);

}} // at::native

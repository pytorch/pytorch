#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/core/DistributionsHelper.h>

namespace at {
namespace native {
namespace {

template<typename scalar_t>
void multinomial_apply(Tensor& result, const Tensor& self, const int64_t n_sample, const bool with_replacement, Generator* generator) {
  auto gen = get_generator_or_default<CPUGenerator>(generator, detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  int64_t n_categories = self.size(-1);
  int64_t n_dist = self.dim() > 1 ? self.size(-2) : 1;

  /* cumulative probability distribution vector */
  Tensor cum_dist = at::empty({n_categories}, self.options());

  const scalar_t * const self_ptr = self.data_ptr<scalar_t>();
  scalar_t * const cum_dist_ptr = cum_dist.data_ptr<scalar_t>();
  int64_t * const result_ptr = result.data_ptr<int64_t>();

  auto self_stride_0 = self.dim() > 1 ? self.stride(-2) : 0;
  auto self_stride_1 = self.stride(-1);

  auto cum_dist_stride_0 = cum_dist.stride(0);

  auto result_dist_stride_0 = result.dim() > 1 ? result.stride(-2) : 0;
  auto result_dist_stride_1 = result.stride(-1);

  for (int64_t i = 0; i < n_dist; i++) {
    /* Get normalized cumulative distribution from prob distribution */
    scalar_t sum = 0;
    scalar_t prev_sum = 0;
    scalar_t val;
    int n_zeros = 0;
    bool lost_precision = false;
    for (int64_t j = 0; j < n_categories; j++) {
      val = self_ptr[i * self_stride_0 + j * self_stride_1];
      TORCH_CHECK(val >= 0, "invalid multinomial distribution (encountering probability entry < 0)");
      TORCH_CHECK(std::isfinite(val), "invalid multinomial distribution (encountering probability entry = infinity or NaN)");

      prev_sum = sum;
      sum += val;
      if (prev_sum == sum && val != 0) {
        // We tried to summarize very large and small numbers and lost precision.
        // We will use extra memory that will hold integer and fractional part of the sum separately.
        lost_precision = true;
        break;
      }
      if (val == 0) {
        n_zeros += 1;
      }
      cum_dist_ptr[j * cum_dist_stride_0] = sum;
    }

    if (!lost_precision) {
      TORCH_CHECK(sum > 0, "invalid multinomial distribution (sum of probabilities <= 0)");
      TORCH_CHECK(with_replacement || (n_categories - n_zeros >= n_sample),
                  "invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)");

      /* normalize cumulative probability distribution so that last val is 1
      i.e. doesn't assume original self row sums to one */
      if ((sum > 0) || ((sum < 1.00001) && (sum > 0.99999))) {
        for (int64_t j = 0; j < n_categories; j++) {
          cum_dist_ptr[j * cum_dist_stride_0] /= sum;
        }
      }

      for (int64_t j = 0; j < n_sample; j++) {
        /* sample a probability mass from a uniform distribution */
        at::uniform_real_distribution<double> uniform(0, 1);
        double uniform_sample = uniform(gen);
        /* Do a binary search for the slot in which the prob falls
        ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
        int left_pointer = 0;
        int right_pointer = n_categories;
        int mid_pointer;
        scalar_t cum_prob;
        int sample_idx;
        /* Make sure the last cumulative distribution bucket sums to 1 */
        cum_dist_ptr[(n_categories - 1) * cum_dist_stride_0] = 1;

        while(right_pointer - left_pointer > 0) {
          mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
          cum_prob = cum_dist_ptr[mid_pointer * cum_dist_stride_0];
          if (cum_prob < uniform_sample) {
            left_pointer = mid_pointer + 1;
          }
          else {
            right_pointer = mid_pointer;
          }
        }
        sample_idx = left_pointer;

        /* store in result tensor (will be incremented for lua compat by wrapper) */
        result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] = sample_idx;

        /* Once a sample is drawn, it cannot be drawn again. ie sample without replacement */
        if (!with_replacement && j < n_sample - 1) {
          /* update cumulative distribution so that sample cannot be drawn again */
          scalar_t diff;
          scalar_t new_val = 0;
          scalar_t sum;

          if (sample_idx != 0) {
            new_val = cum_dist_ptr[(sample_idx - 1) * cum_dist_stride_0];
          }
          /* marginal cumulative mass (i.e. original probability) of sample */
          diff = cum_dist_ptr[sample_idx * cum_dist_stride_0] - new_val;
          /* new sum of marginals is not one anymore... */
          sum = 1.0 - diff;
          for (int64_t k = 0; k < n_categories; k++) {
            new_val = cum_dist_ptr[k * cum_dist_stride_0];
            if (k >= sample_idx) {
              /* remove sampled probability mass from later cumulative probabilities */
              new_val -= diff;
            }
            /* make total marginals sum to one */
            new_val /= sum;
            cum_dist_ptr[k * cum_dist_stride_0] = new_val;
          }
        }
      }
    } else {
      AT_WARN("To calculate cumulative distribution pytorch will use extra memory");
      // We will use second Tensor cum_dist_high to hold integer part of the cum_dist
      // cum_dist will be used to hold fractional part
      Tensor cum_dist_high = at::empty({n_categories}, self.options());
      scalar_t * const cum_dist_high_ptr = cum_dist_high.data_ptr<scalar_t>();
      scalar_t * const cum_dist_low_ptr = cum_dist.data_ptr<scalar_t>();

      scalar_t sum_high = 0;
      scalar_t sum_low = 0;
      scalar_t val;
      for (int64_t j = 0; j < n_categories; j++) {
        val = self_ptr[i * self_stride_0 + j * self_stride_1];
        TORCH_CHECK(val >= 0, "invalid multinomial distribution (encountering probability entry < 0)");
        TORCH_CHECK(std::isfinite(val), "invalid multinomial distribution (encountering probability entry = infinity or NaN)");

        scalar_t h = (scalar_t)((long)val);
        scalar_t l = val - h;
        sum_high += h;
        sum_low += l;
        if (val == 0) {
          n_zeros += 1;
        }
        cum_dist_high_ptr[j * cum_dist_stride_0] = sum_high;
        cum_dist_low_ptr[j * cum_dist_stride_0] = sum_low;
      }

      TORCH_CHECK((sum_high + sum_low) > 0, "invalid multinomial distribution (sum of probabilities <= 0)");
      TORCH_CHECK(with_replacement || (n_categories - n_zeros >= n_sample),
                  "invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)");

      for (int64_t j = 0; j < n_sample; j++) {
        at::uniform_real_distribution<double> uniform(0, 1);
        double uniform_sample = uniform(gen);
        /* Do a binary search for the slot in which the prob falls
        ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
        int left_pointer = 0;
        int right_pointer = n_categories;
        int mid_pointer;
        scalar_t cum_prob_high;
        scalar_t cum_prob_low;
        int sample_idx;

        while(right_pointer - left_pointer > 0) {
          mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
          cum_prob_high = cum_dist_high_ptr[mid_pointer * cum_dist_stride_0];
          cum_prob_low = cum_dist_low_ptr[mid_pointer * cum_dist_stride_0];
          scalar_t diff = cum_prob_high - uniform_sample * sum_high + cum_prob_low - uniform_sample * sum_low;
          if (diff < 0) {
            left_pointer = mid_pointer + 1;
          }
          else {
            right_pointer = mid_pointer;
          }
        }
        sample_idx = left_pointer;

        /* store in result tensor (will be incremented for lua compat by wrapper) */
        result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] = sample_idx;

        /* Once a sample is drawn, it cannot be drawn again. ie sample without replacement */
        if (!with_replacement && j < n_sample - 1) {
          /* update cumulative distribution so that sample cannot be drawn again */
          scalar_t diff_high;
          scalar_t diff_low;
          scalar_t new_val_high = 0;
          scalar_t new_val_low = 0;

          if (sample_idx != 0) {
            new_val_high = cum_dist_high_ptr[(sample_idx - 1) * cum_dist_stride_0];
            new_val_low = cum_dist_low_ptr[(sample_idx - 1) * cum_dist_stride_0];
          }
          /* marginal cumulative mass (i.e. original probability) of sample */
          diff_high = cum_dist_high_ptr[sample_idx * cum_dist_stride_0] - new_val_high;
          diff_low = cum_dist_low_ptr[sample_idx * cum_dist_stride_0] - new_val_low;

          sum_high -= diff_high;
          sum_low -= diff_low;
          for (int64_t k = 0; k < n_categories; k++) {
            new_val_high = cum_dist_high_ptr[k * cum_dist_stride_0];
            new_val_low = cum_dist_low_ptr[k * cum_dist_stride_0];
            if (k >= sample_idx) {
              /* remove sampled probability mass from later cumulative probabilities */
              new_val_high -= diff_high;
              new_val_low -= diff_low;
            }
            cum_dist_high_ptr[k * cum_dist_stride_0] = new_val_high;
            cum_dist_low_ptr[k * cum_dist_stride_0] = new_val_low;
          }
        }
      }
    }


  }
}

static void multinomial_kernel_impl(Tensor& result, const Tensor& self, const int64_t n_sample, const bool with_replacement, Generator *gen) {
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "multinomial", [&] {
    multinomial_apply<scalar_t>(result, self, n_sample, with_replacement, gen);
  });
}

}

REGISTER_DISPATCH(multinomial_stub, &multinomial_kernel_impl);

}
}

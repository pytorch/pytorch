#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.c"
#else

#define DEBUG 1

TH_API void THTensor_(random)(THTensor *self)
{
#if defined(TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY(real, self, *self_data = (unsigned char)(THRandom_random() % (UCHAR_MAX+1)););
#elif defined(TH_REAL_IS_CHAR)
  TH_TENSOR_APPLY(real, self, *self_data = (char)(THRandom_random() % (CHAR_MAX+1)););
#elif defined(TH_REAL_IS_SHORT)
  TH_TENSOR_APPLY(real, self, *self_data = (short)(THRandom_random() % (SHRT_MAX+1)););
#elif defined(TH_REAL_IS_INT)
  TH_TENSOR_APPLY(real, self, *self_data = (int)(THRandom_random() % (INT_MAX+1UL)););
#elif defined(TH_REAL_IS_LONG)
  TH_TENSOR_APPLY(real, self, *self_data = (long)(THRandom_random() % (LONG_MAX+1UL)););
#elif defined(TH_REAL_IS_FLOAT)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_random() % ((1UL << FLT_MANT_DIG)+1)););
#elif defined(TH_REAL_IS_DOUBLE)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_random() % ((1UL << DBL_MANT_DIG)+1)););
#else
#error "Unknown type"
#endif
}

TH_API void THTensor_(geometric)(THTensor *self, double p)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_geometric(p););
}

TH_API void THTensor_(bernoulli)(THTensor *self, double p)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_bernoulli(p););
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(uniform)(THTensor *self, double a, double b)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_uniform(a, b););
}

TH_API void THTensor_(normal)(THTensor *self, double mean, double stdv)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_normal(mean, stdv););
}

TH_API void THTensor_(exponential)(THTensor *self, double lambda)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_exponential(lambda););
}

TH_API void THTensor_(cauchy)(THTensor *self, double median, double sigma)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_cauchy(median, sigma););
}

TH_API void THTensor_(logNormal)(THTensor *self, double mean, double stdv)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_logNormal(mean, stdv););
}

// update a row of cumulative distribution from prob distribution
// @param:
//    cum_dist: cumulative probability distribution
//    prob_dist: probability distribution
//    row: row of the cum_distr to be updated   
static void THTensor_(update_cum_row)(THTensor* cum_dist, THTensor* prob_dist, int row)
{
  long n_categories = THTensor_(size)(cum_dist, 1);
  real sum = 0;
  int j;
  for (j=0; j<n_categories; j++)
  {
    sum += THStorage_(get)(prob_dist->storage, prob_dist->storageOffset+row*prob_dist->stride[0]+j*prob_dist->stride[1]);
    THStorage_(set)(cum_dist->storage, cum_dist->storageOffset+row*cum_dist->stride[0]+j*cum_dist->stride[1], sum);
  }
}


// generate cumulative distribution from probability distribution
// @param:
//   prob_dist: probability distribution (rows sum to one)
static THTensor* THTensor_(generate_cum_matrix)(THTensor *prob_dist)
{
  THTensor *cum_dist = THTensor_(new)();
  THTensor_(resizeAs)(cum_dist, prob_dist);

  // Get normalized cumulative distribution from prob distribution
  int i;
  for (i=0; i<THTensor_(size)(cum_dist, 0); i++)
  {
    THTensor_(update_cum_row)(cum_dist, prob_dist, i);
  }
  return cum_dist;
}

// Do a binary search for the slot in which the prob falls
// @params:
//     cum_width: width of the cum_distr
//     row: the row of the cum_distr matrix to search where the prob falls
//     uniform_sample: a sample from an uniform distribution U ~ [0, 1]
// @return: the slot in which the prob falls, 
//     ie cum_distr[row][slot-1] < prob < cum_distr[row][slot]
static int THTensor_(binarySearch)(THTensor* cum_dist, int row, real uniform_sample)
{
    int left_pointer = 0;
    int right_pointer = (int)THTensor_(size)(cum_dist, 1);
    int mid_pointer;
    real cum_prob;
    while(right_pointer - left_pointer > 0)
    {
        mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
        cum_prob = THStorage_(get)(cum_dist->storage, cum_dist->storageOffset+row*cum_dist->stride[0]+mid_pointer*cum_dist->stride[1]);
        printf("pointer, tmp : %d %d %d : %f > %f", left_pointer, right_pointer, mid_pointer, uniform_sample, cum_prob);
        if (cum_prob < uniform_sample) 
        {
          printf(" search right\n");
          left_pointer = mid_pointer + 1;
        }
        else
        {
          printf(" search left\n");
          right_pointer = mid_pointer;
        }
    }
    
    printf("return %d \n", left_pointer);
    return left_pointer;
}


// Generate a subset of random samples from the prob_dist
// @param:
//     h: height of multinomial matrix, each row in matrix represents an individual experiment
//     prob_width: width of prob_distr
//     num_samples: number of samples to sample from the multinomial distribution
//     with_replacement: 1 for true, 0 for false
// @return:
//     a sample of experts from the multinomial probability distribution
TH_API void THTensor_(multinomial)(THLongTensor *self, THTensor *prob_dist, int n_sample, int with_replacement)
{
  printf("n_sample : %d\n", n_sample);
  THTensor* cum_dist = THTensor_(generate_cum_matrix)(prob_dist);    

  // Generate 2d samples randomly from a uniform distribution ~ [0, 1]
  THTensor* uniform_samples = THTensor_(new)();
  long size[2] = {THTensor_(size)(cum_dist, 0), (long)n_sample};
  THLongStorage* sample_size = THLongStorage_newWithData(size, 2);
  THTensor_(rand)(uniform_samples, sample_size);
  
  // multinomial samples
  THLongTensor_resize(self, sample_size, NULL);
  
  // Allows drawn sample to be placed back into the pool for drawing again. ie with replacement
  int i,j;
  if (with_replacement)
  {
    for (i=0; i<THTensor_(size)(cum_dist, 0); i++)
    {
      for (j=0; j<n_sample; j++)
      {
        real uniform_sample = THStorage_(get)(uniform_samples->storage, uniform_samples->storageOffset+i*uniform_samples->stride[0]+j*uniform_samples->stride[1]);
        // increment sample index for lua compat
        THLongStorage_set(self->storage, self->storageOffset+i*self->stride[0]+j*self->stride[1], 1 + THTensor_(binarySearch)(cum_dist, i, uniform_sample));
        if (DEBUG)
        {
          printf("(%d, %d): random_sample %f in slot %ld \n", i, j, \
          uniform_sample, THLongStorage_get(self->storage, self->storageOffset+i*self->stride[0]+j*self->stride[1]));
        }                                     
      }
    }
  }
  /**
  // Once sample is drawn, it cannot be drawn again. ie sample without replacement
  else
  {
      for (i=0; i<h; i++)
      {   
          for (j=0; j<num_samples; j++)
          {
              if (DEBUG)
              {
                  printf("==before==\n");
                  printDouble(h, prob_width+1, cum_distr);
              } 
          
              int sample = binarySearch(cum_distr, prob_width + 1, i, random_samples[i][j]);
              // increase all the sample index by 1
              self[i][j] = sample + 1;
              prob_distr[i][sample] = 0;
              update_cum_row(cum_distr, prob_distr, i, prob_width);
              
              if (DEBUG)
              {
                  printf("==after==");
                  printf("(%d, %d): random_sample %f in slot %d \n", i, j, \
                  random_samples[i][j], self[i][j]);
                  printDouble(h, prob_width, prob_distr);
                  printDouble(h, prob_width+1, cum_distr);
              } 
          }
      }
  }**/
  
  THTensor_(free)(cum_dist);
  THTensor_(free)(uniform_samples);
}

#endif

#if defined(TH_REAL_IS_LONG)
TH_API void THTensor_(getRNGState)(THTensor *self)
{
  unsigned long *data;
  long *offset;
  long *left;

  THTensor_(resize1d)(self,626);
  data = (unsigned long *)THTensor_(data)(self);
  offset = (long *)data+624;
  left = (long *)data+625;

  THRandom_getState(data,offset,left);
}

TH_API void THTensor_(setRNGState)(THTensor *self)
{
  unsigned long *data;
  long *offset;
  long *left;

  THArgCheck(THTensor_(nElement)(self) == 626, 1, "state should have 626 elements");
  data = (unsigned long *)THTensor_(data)(self);
  offset = (long *)(data+624);
  left = (long *)(data+625);

  THRandom_setState(data,*offset,*left);
}

#endif

#endif

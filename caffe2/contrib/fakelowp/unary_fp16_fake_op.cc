#include "unary_fp16_fake_op.h"
#include <fbgemm/FbgemmConvert.h>
#include "caffe2/contrib/fakelowp/fp16_fma.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace fake_fp16 {
auto sig_lut = std::vector<at::Half>{
    0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f, 0.0000e+00f,
    0.0000e+00f, 0.0000e+00f, 5.9605e-08f, 5.9605e-08f, 5.9605e-08f,
    5.9605e-08f, 5.9605e-08f, 5.9605e-08f, 5.9605e-08f, 5.9605e-08f,
    5.9605e-08f, 5.9605e-08f, 1.1921e-07f, 1.1921e-07f, 1.1921e-07f,
    1.1921e-07f, 1.7881e-07f, 1.7881e-07f, 1.7881e-07f, 2.3842e-07f,
    2.3842e-07f, 2.3842e-07f, 2.9802e-07f, 2.9802e-07f, 3.5763e-07f,
    4.1723e-07f, 4.7684e-07f, 4.7684e-07f, 5.3644e-07f, 6.5565e-07f,
    7.1526e-07f, 7.7486e-07f, 8.9407e-07f, 9.5367e-07f, 1.0729e-06f,
    1.1921e-06f, 1.3709e-06f, 1.4901e-06f, 1.6689e-06f, 1.8477e-06f,
    2.0862e-06f, 2.3246e-06f, 2.6226e-06f, 2.9206e-06f, 3.2187e-06f,
    3.6359e-06f, 4.0531e-06f, 4.4703e-06f, 5.0068e-06f, 5.6028e-06f,
    6.2585e-06f, 6.9737e-06f, 7.7486e-06f, 8.6427e-06f, 9.6560e-06f,
    1.0788e-05f, 1.2040e-05f, 1.3411e-05f, 1.4961e-05f, 1.6689e-05f,
    1.8597e-05f, 2.0742e-05f, 2.3186e-05f, 2.5868e-05f, 2.8849e-05f,
    3.2187e-05f, 3.5882e-05f, 4.0054e-05f, 4.4644e-05f, 4.9829e-05f,
    5.5552e-05f, 6.1989e-05f, 6.9141e-05f, 7.7128e-05f, 8.6069e-05f,
    9.6023e-05f, 1.0711e-04f, 1.1951e-04f, 1.3328e-04f, 1.4865e-04f,
    1.6594e-04f, 1.8501e-04f, 2.0647e-04f, 2.3031e-04f, 2.5702e-04f,
    2.8658e-04f, 3.1972e-04f, 3.5667e-04f, 3.9792e-04f, 4.4370e-04f,
    4.9496e-04f, 5.5218e-04f, 6.1607e-04f, 6.8712e-04f, 7.6675e-04f,
    8.5497e-04f, 9.5367e-04f, 1.0643e-03f, 1.1864e-03f, 1.3237e-03f,
    1.4763e-03f, 1.6470e-03f, 1.8368e-03f, 2.0485e-03f, 2.2850e-03f,
    2.5482e-03f, 2.8419e-03f, 3.1700e-03f, 3.5343e-03f, 3.9406e-03f,
    4.3945e-03f, 4.9019e-03f, 5.4626e-03f, 6.0921e-03f, 6.7902e-03f,
    7.5722e-03f, 8.4381e-03f, 9.4070e-03f, 1.0483e-02f, 1.1681e-02f,
    1.3008e-02f, 1.4488e-02f, 1.6144e-02f, 1.7975e-02f, 2.0004e-02f,
    2.2263e-02f, 2.4780e-02f, 2.7557e-02f, 3.0655e-02f, 3.4058e-02f,
    3.7872e-02f, 4.2053e-02f, 4.6692e-02f, 5.1819e-02f, 5.7434e-02f,
    6.3660e-02f, 7.0496e-02f, 7.8003e-02f, 8.6243e-02f, 9.5276e-02f,
    1.0516e-01f, 1.1591e-01f, 1.2756e-01f, 1.4026e-01f, 1.5393e-01f,
    1.6882e-01f, 1.8469e-01f, 2.0178e-01f, 2.1997e-01f, 2.3926e-01f,
    2.5977e-01f, 2.8125e-01f, 3.0396e-01f, 3.2764e-01f, 3.5205e-01f,
    3.7744e-01f, 4.0356e-01f, 4.3018e-01f, 4.5703e-01f, 4.8438e-01f,
    5.1172e-01f, 5.3906e-01f, 5.6592e-01f, 5.9277e-01f, 6.1865e-01f,
    6.4404e-01f, 6.6895e-01f, 6.9287e-01f, 7.1533e-01f, 7.3730e-01f,
    7.5781e-01f, 7.7734e-01f, 7.9590e-01f, 8.1299e-01f, 8.2910e-01f,
    8.4375e-01f, 8.5791e-01f, 8.7061e-01f, 8.8232e-01f, 8.9355e-01f,
    9.0332e-01f, 9.1260e-01f, 9.2090e-01f, 9.2822e-01f, 9.3555e-01f,
    9.4189e-01f, 9.4727e-01f, 9.5264e-01f, 9.5752e-01f, 9.6143e-01f,
    9.6533e-01f, 9.6875e-01f, 9.7217e-01f, 9.7461e-01f, 9.7754e-01f,
    9.7949e-01f, 9.8193e-01f, 9.8340e-01f, 9.8535e-01f, 9.8682e-01f,
    9.8828e-01f, 9.8926e-01f, 9.9023e-01f, 9.9121e-01f, 9.9219e-01f,
    9.9316e-01f, 9.9365e-01f, 9.9463e-01f, 9.9512e-01f, 9.9561e-01f,
    9.9609e-01f, 9.9658e-01f, 9.9658e-01f, 9.9707e-01f, 9.9756e-01f,
    9.9756e-01f, 9.9805e-01f, 9.9805e-01f, 9.9854e-01f, 9.9854e-01f,
    9.9854e-01f, 9.9902e-01f, 9.9902e-01f, 9.9902e-01f, 9.9902e-01f,
    9.9902e-01f, 9.9951e-01f, 9.9951e-01f, 9.9951e-01f, 9.9951e-01f,
    9.9951e-01f, 9.9951e-01f, 9.9951e-01f, 9.9951e-01f, 9.9951e-01f,
    9.9951e-01f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f,
    1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f,
    1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f,
    1.0000e+00f, 1.0000e+00f};

at::Half CalcSigmoidByLUT(at::Half x) {
  at::Half a = -18.0;
  at::Half b = 10.0;
  int nBins = 256;

  at::Half delta = (b - a) / (at::Half)nBins;
  at::Half one_over_delta = 1 / delta;
  at::Half a_one_over_delta = a * one_over_delta;

  // Clamp the input in the range of a to b
  if (x < a) {
    x = a;
  }

  if (x > b) {
    x = b;
  }

  at::Half bin_calc = std::fma(x, one_over_delta, -a_one_over_delta);

  uint32_t bin = bin_calc < 0 ? 0 : (uint32_t)floor(bin_calc);
  // Clamp bin to SIGMOID_KNOT_LUT_SIZE-2, to have valid LUT access i.e. b+1 =
  // 255 (for LUT size of 256)
  if (bin > 254) {
    bin = 254;
  }
  // Use MAC bin_x = a + delta * at::Half(bin);
  at::Half bin_x = std::fma(delta, at::Half(bin), a);

  at::Half p = at::Half(x - bin_x) * one_over_delta;

  at::Half res1 = sig_lut[bin + 1] * p;
  // Use MAC res2 = (1 - p) * lut[bin] = -p * lut[bin] + lut[bin]
  at::Half res2 = std::fma(-p, sig_lut[bin], sig_lut[bin]);

  return at::Half(res1 + res2);
}

const int TANH_LINEAR_MAX_VALUE = 10048;
const int TANH_ASYMPTOTE_MIN_VALUE = 17538;

static float tanh_lut[] = {
    at::Half(0.02831274f), at::Half(0.02928850f), at::Half(0.03026419f),
    at::Half(0.03123983f), at::Half(0.03319093f), at::Half(0.03514177f),
    at::Half(0.03709235f), at::Half(0.03904264f), at::Half(0.04099264f),
    at::Half(0.04294232f), at::Half(0.04489168f), at::Half(0.04684070f),
    at::Half(0.04878936f), at::Half(0.05073764f), at::Half(0.05268555f),
    at::Half(0.05463305f), at::Half(0.05658013f), at::Half(0.05852679f),
    at::Half(0.06047300f), at::Half(0.06241875f), at::Half(0.06630881f),
    at::Half(0.07019686f), at::Half(0.07408277f), at::Half(0.07796644f),
    at::Half(0.08184774f), at::Half(0.08572657f), at::Half(0.08960279f),
    at::Half(0.09347630f), at::Half(0.09734699f), at::Half(0.10121473f),
    at::Half(0.10507942f), at::Half(0.10894093f), at::Half(0.11279916f),
    at::Half(0.11665399f), at::Half(0.12050531f), at::Half(0.12435300f),
    at::Half(0.13203707f), at::Half(0.13970530f), at::Half(0.14735681f),
    at::Half(0.15499073f), at::Half(0.16260618f), at::Half(0.17020231f),
    at::Half(0.17777826f), at::Half(0.18533320f), at::Half(0.19286629f),
    at::Half(0.20037672f), at::Half(0.20786367f), at::Half(0.21532634f),
    at::Half(0.22276395f), at::Half(0.23017571f), at::Half(0.23756087f),
    at::Half(0.24491866f), at::Half(0.25954921f), at::Half(0.27406159f),
    at::Half(0.28845021f), at::Half(0.30270973f), at::Half(0.31683500f),
    at::Half(0.33082112f), at::Half(0.34466340f), at::Half(0.35835740f),
    at::Half(0.37189891f), at::Half(0.38528397f), at::Half(0.39850884f),
    at::Half(0.41157006f), at::Half(0.42446437f), at::Half(0.43718879f),
    at::Half(0.44974055f), at::Half(0.46211716f), at::Half(0.48633602f),
    at::Half(0.50982997f), at::Half(0.53258729f), at::Half(0.55459972f),
    at::Half(0.57586239f), at::Half(0.59637356f), at::Half(0.61613443f),
    at::Half(0.63514895f), at::Half(0.65342359f), at::Half(0.67096707f),
    at::Half(0.68779021f), at::Half(0.70390560f), at::Half(0.71932750f),
    at::Half(0.73407152f), at::Half(0.74815447f), at::Half(0.76159416f),
    at::Half(0.78661881f), at::Half(0.80930107f), at::Half(0.82980191f),
    at::Half(0.84828364f), at::Half(0.86490662f), at::Half(0.87982670f),
    at::Half(0.89319334f), at::Half(0.90514825f), at::Half(0.91582454f),
    at::Half(0.92534623f), at::Half(0.93382804f), at::Half(0.94137554f),
    at::Half(0.94808529f), at::Half(0.95404526f), at::Half(0.95933529f),
    at::Half(0.96402758f), at::Half(0.97187275f), at::Half(0.97802611f),
    at::Half(0.98284503f), at::Half(0.98661430f), at::Half(0.98955975f),
    at::Half(0.99185972f), at::Half(0.99365463f), at::Half(0.99505475f),
    at::Half(0.99614653f), at::Half(0.99699764f), at::Half(0.99766098f),
    at::Half(0.99817790f), at::Half(0.99858066f), at::Half(0.99889444f),
    at::Half(0.99913889f), at::Half(0.99932930f), at::Half(0.99959315f),
    at::Half(0.99975321f), at::Half(0.99985031f), at::Half(0.99990920f),
    at::Half(0.99994493f), at::Half(0.99996660f), at::Half(0.99997974f),
    at::Half(0.99998771f), at::Half(0.99999255f), at::Half(0.99999548f),
    at::Half(0.99999726f), at::Half(0.99999834f)};

static float tanh_error_lut[] = {
    at::Half(0.00001525f), at::Half(0.00001525f), at::Half(0.00001524f),
    at::Half(0.00003049f), at::Half(0.00003048f), at::Half(0.00003048f),
    at::Half(0.00003047f), at::Half(0.00003047f), at::Half(0.00003046f),
    at::Half(0.00003046f), at::Half(0.00003045f), at::Half(0.00003045f),
    at::Half(0.00003044f), at::Half(0.00003044f), at::Half(0.00003043f),
    at::Half(0.00003042f), at::Half(0.00003042f), at::Half(0.00003041f),
    at::Half(0.00003040f), at::Half(0.00006078f), at::Half(0.00006075f),
    at::Half(0.00006072f), at::Half(0.00006068f), at::Half(0.00006065f),
    at::Half(0.00006061f), at::Half(0.00006057f), at::Half(0.00006052f),
    at::Half(0.00006048f), at::Half(0.00006043f), at::Half(0.00006039f),
    at::Half(0.00006034f), at::Half(0.00006028f), at::Half(0.00006023f),
    at::Half(0.00006018f), at::Half(0.00006012f), at::Half(0.00012006f),
    at::Half(0.00011982f), at::Half(0.00011955f), at::Half(0.00011928f),
    at::Half(0.00011899f), at::Half(0.00011869f), at::Half(0.00011837f),
    at::Half(0.00011805f), at::Half(0.00011770f), at::Half(0.00011735f),
    at::Half(0.00011698f), at::Half(0.00011660f), at::Half(0.00011621f),
    at::Half(0.00011581f), at::Half(0.00011539f), at::Half(0.00011497f),
    at::Half(0.00022860f), at::Half(0.00022676f), at::Half(0.00022482f),
    at::Half(0.00022281f), at::Half(0.00022071f), at::Half(0.00021853f),
    at::Half(0.00021629f), at::Half(0.00021397f), at::Half(0.00021159f),
    at::Half(0.00020914f), at::Half(0.00020664f), at::Half(0.00020408f),
    at::Half(0.00020147f), at::Half(0.00019882f), at::Half(0.00019612f),
    at::Half(0.00019338f), at::Half(0.00037842f), at::Half(0.00036709f),
    at::Half(0.00035558f), at::Half(0.00034394f), at::Half(0.00033223f),
    at::Half(0.00032049f), at::Half(0.00030876f), at::Half(0.00029710f),
    at::Half(0.00028554f), at::Half(0.00027412f), at::Half(0.00026286f),
    at::Half(0.00025180f), at::Half(0.00024097f), at::Half(0.00023038f),
    at::Half(0.00022005f), at::Half(0.00021000f), at::Half(0.00039101f),
    at::Half(0.00035441f), at::Half(0.00032033f), at::Half(0.00028878f),
    at::Half(0.00025973f), at::Half(0.00023313f), at::Half(0.00020885f),
    at::Half(0.00018680f), at::Half(0.00016682f), at::Half(0.00014878f),
    at::Half(0.00013253f), at::Half(0.00011793f), at::Half(0.00010484f),
    at::Half(0.00009312f), at::Half(0.00008266f), at::Half(0.00007332f),
    at::Half(0.00012258f), at::Half(0.00009615f), at::Half(0.00007530f),
    at::Half(0.00005889f), at::Half(0.00004602f), at::Half(0.00003594f),
    at::Half(0.00002805f), at::Half(0.00002188f), at::Half(0.00001706f),
    at::Half(0.00001330f), at::Half(0.00001036f), at::Half(0.00000808f),
    at::Half(0.00000629f), at::Half(0.00000490f), at::Half(0.00000382f),
    at::Half(0.00000298f), at::Half(0.00000000f), at::Half(0.00000000f),
    at::Half(0.00000000f), at::Half(0.00000000f), at::Half(0.00000000f),
    at::Half(0.00000000f), at::Half(0.00000000f), at::Half(0.00000000f),
    at::Half(0.00000000f), at::Half(0.00000000f), at::Half(0.00000000f),
    at::Half(0.00000000f), at::Half(0.00000000f)};

at::Half CalcTanhByLUT(at::Half input) {
  uint16_t InputInU16_temp;
  uint16_t InputInU16;
  int mask = 0x7FFF;
  uint16_t sign_bit;
  float unit = 1.0;
  float index;
  at::Half err_f16;
  at::Half output = at::Half(0.0f);

  /* Extracting bits 9-15 of f16 input to get the LUT index */
  InputInU16_temp = (*((uint16_t*)&input));
  sign_bit = InputInU16_temp & 0x8000;
  InputInU16 = InputInU16_temp & mask; // positive number
  if (InputInU16 < TANH_LINEAR_MAX_VALUE) {
    output = input;
  } else if (InputInU16 >= TANH_ASYMPTOTE_MIN_VALUE) {
    output = unit;
  } else {
    index = ((InputInU16 - TANH_LINEAR_MAX_VALUE) % 64);
    err_f16 =
        at::Half(tanh_error_lut[(InputInU16 - TANH_LINEAR_MAX_VALUE) / 64]);
    output = at::Half(tanh_lut[(InputInU16 - TANH_LINEAR_MAX_VALUE) / 64]);

    output = at::Half(std::fma(err_f16, index, output));
  }
  uint16_t outputInU16_temp = (*((uint16_t*)&output)) | sign_bit;
  output = (*((at::Half*)&outputInU16_temp));
  return output;
}

at::Half CalcTanhByPolynomial(at::Half input) {
  static const at::Half aCoefficient[64] = {
      at::Half(-0.423340f), at::Half(-0.352783f), at::Half(-0.411377f),
      at::Half(-0.284424f), at::Half(-0.335938f), at::Half(-0.333740f),
      at::Half(-0.333252f), at::Half(-0.332275f), at::Half(-0.333252f),
      at::Half(-0.333252f), at::Half(-0.333252f), at::Half(-0.333252f),
      at::Half(-0.333252f), at::Half(-0.333252f), at::Half(-0.333252f),
      at::Half(-0.333252f), at::Half(-0.333008f), at::Half(-0.333008f),
      at::Half(-0.332764f), at::Half(-0.332275f), at::Half(-0.331055f),
      at::Half(-0.329346f), at::Half(-0.325195f), at::Half(-0.317383f),
      at::Half(-0.301758f), at::Half(-0.273438f), at::Half(-0.219360f),
      at::Half(-0.136108f), at::Half(-0.018677f), at::Half(0.080872f),
      at::Half(0.107056f),  at::Half(0.063110f),  at::Half(0.017731f),
      at::Half(0.002533f),  at::Half(0.000147f),  at::Half(0.000003f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f)};
  static const at::Half bCoefficient[64] = {
      at::Half(0.000004f),  at::Half(0.000002f),  at::Half(0.000017f),
      at::Half(-0.000016f), at::Half(0.000001f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(-0.000001f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000001f), at::Half(-0.000002f), at::Half(-0.000007f),
      at::Half(-0.000020f), at::Half(-0.000054f), at::Half(-0.000158f),
      at::Half(-0.000433f), at::Half(-0.001253f), at::Half(-0.003410f),
      at::Half(-0.009712f), at::Half(-0.025681f), at::Half(-0.068665f),
      at::Half(-0.162354f), at::Half(-0.346680f), at::Half(-0.566406f),
      at::Half(-0.640137f), at::Half(-0.439941f), at::Half(-0.161255f),
      at::Half(-0.030548f), at::Half(-0.002459f), at::Half(-0.000061f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f)};
  static const at::Half cCoefficient[64] = {
      at::Half(1.000000f), at::Half(1.000000f), at::Half(1.000000f),
      at::Half(1.000000f), at::Half(1.000000f), at::Half(1.000000f),
      at::Half(1.000000f), at::Half(1.000000f), at::Half(1.000000f),
      at::Half(1.000000f), at::Half(1.000000f), at::Half(1.000000f),
      at::Half(1.000000f), at::Half(1.000000f), at::Half(1.000000f),
      at::Half(1.000000f), at::Half(1.000000f), at::Half(1.000000f),
      at::Half(1.000000f), at::Half(1.000000f), at::Half(1.000000f),
      at::Half(1.000000f), at::Half(1.000000f), at::Half(1.000000f),
      at::Half(1.000977f), at::Half(1.003906f), at::Half(1.014648f),
      at::Half(1.050781f), at::Half(1.147461f), at::Half(1.309570f),
      at::Half(1.378906f), at::Half(1.073242f), at::Half(0.500488f),
      at::Half(0.124329f), at::Half(0.013718f), at::Half(0.000464f),
      at::Half(0.000004f), at::Half(0.000000f), at::Half(0.000000f),
      at::Half(0.000000f), at::Half(0.000000f), at::Half(0.000000f),
      at::Half(0.000000f), at::Half(0.000000f), at::Half(0.000000f),
      at::Half(0.000000f), at::Half(0.000000f), at::Half(0.000000f),
      at::Half(0.000000f), at::Half(0.000000f), at::Half(0.000000f),
      at::Half(0.000000f), at::Half(0.000000f), at::Half(0.000000f),
      at::Half(0.000000f), at::Half(0.000000f), at::Half(0.000000f),
      at::Half(0.000000f), at::Half(0.000000f), at::Half(0.000000f),
      at::Half(0.000000f), at::Half(0.000000f)};
  static const at::Half dCoefficient[64] = {
      at::Half(-0.000000f), at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(-0.000000f), at::Half(0.000000f),  at::Half(0.000000f),
      at::Half(0.000000f),  at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000000f), at::Half(-0.000000f),
      at::Half(-0.000000f), at::Half(-0.000001f), at::Half(-0.000008f),
      at::Half(-0.000045f), at::Half(-0.000237f), at::Half(-0.001252f),
      at::Half(-0.005722f), at::Half(-0.022766f), at::Half(-0.062866f),
      at::Half(-0.084229f), at::Half(0.071167f),  at::Half(0.466064f),
      at::Half(0.828125f),  at::Half(0.974121f),  at::Half(0.998535f),
      at::Half(0.999512f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f),  at::Half(1.000000f),  at::Half(1.000000f),
      at::Half(1.000000f)};

  int16_t temp = *((unsigned short*)(&input));
  int16_t index = ((temp & 0x7E00) >> 9); // extract bits 9..14

  // Because tanh is anti-symmetric, we can perform the operation for abs(t_2)
  // and then multiply the result by -1 in case the number is negative.
  at::Half absInput = (input < 0) ? (input * at::Half(-1)) : input;

  at::Half a = aCoefficient[index];
  at::Half b = bCoefficient[index];
  at::Half c = cCoefficient[index];
  at::Half d = dCoefficient[index];

  b = b + a * absInput;
  c = c + b * absInput;
  at::Half tanhResult = d + c * absInput;
  tanhResult =
      (input < 0) ? tanhResult * -1 : tanhResult; // tanh is anti-symmetric

  return tanhResult;
}

static const float swishLutKnot[] = {
    -0.000000025618f, -0.000000027492f, -0.000000029503f, -0.000000031660f,
    -0.000000033974f, -0.000000036457f, -0.000000039121f, -0.000000041979f,
    -0.000000045045f, -0.000000048335f, -0.000000051864f, -0.000000055650f,
    -0.000000059711f, -0.000000064068f, -0.000000068742f, -0.000000073756f,
    -0.000000079134f, -0.000000084903f, -0.000000091091f, -0.000000097729f,
    -0.000000104849f, -0.000000112487f, -0.000000120678f, -0.000000129464f,
    -0.000000138888f, -0.000000148995f, -0.000000159835f, -0.000000171461f,
    -0.000000183930f, -0.000000197302f, -0.000000211643f, -0.000000227023f,
    -0.000000243516f, -0.000000261203f, -0.000000280171f, -0.000000300510f,
    -0.000000322320f, -0.000000345707f, -0.000000370785f, -0.000000397675f,
    -0.000000426507f, -0.000000457421f, -0.000000490568f, -0.000000526106f,
    -0.000000564209f, -0.000000605061f, -0.000000648857f, -0.000000695811f,
    -0.000000746149f, -0.000000800113f, -0.000000857963f, -0.000000919978f,
    -0.000000986455f, -0.000001057716f, -0.000001134101f, -0.000001215978f,
    -0.000001303740f, -0.000001397807f, -0.000001498630f, -0.000001606692f,
    -0.000001722509f, -0.000001846635f, -0.000001979663f, -0.000002122227f,
    -0.000002275009f, -0.000002438735f, -0.000002614186f, -0.000002802195f,
    -0.000003003658f, -0.000003219530f, -0.000003450837f, -0.000003698675f,
    -0.000003964218f, -0.000004248724f, -0.000004553538f, -0.000004880101f,
    -0.000005229955f, -0.000005604750f, -0.000006006252f, -0.000006436353f,
    -0.000006897075f, -0.000007390585f, -0.000007919199f, -0.000008485397f,
    -0.000009091833f, -0.000009741347f, -0.000010436975f, -0.000011181970f,
    -0.000011979807f, -0.000012834208f, -0.000013749153f, -0.000014728900f,
    -0.000015778001f, -0.000016901329f, -0.000018104095f, -0.000019391869f,
    -0.000020770612f, -0.000022246698f, -0.000023826941f, -0.000025518629f,
    -0.000027329555f, -0.000029268051f, -0.000031343023f, -0.000033563995f,
    -0.000035941147f, -0.000038485362f, -0.000041208271f, -0.000044122307f,
    -0.000047240758f, -0.000050577824f, -0.000054148682f, -0.000057969547f,
    -0.000062057748f, -0.000066431796f, -0.000071111470f, -0.000076117902f,
    -0.000081473662f, -0.000087202860f, -0.000093331247f, -0.000099886327f,
    -0.000106897471f, -0.000114396042f, -0.000122415530f, -0.000130991691f,
    -0.000140162700f, -0.000149969309f, -0.000160455017f, -0.000171666254f,
    -0.000183652574f, -0.000196466858f, -0.000210165536f, -0.000224808818f,
    -0.000240460939f, -0.000257190429f, -0.000275070386f, -0.000294178776f,
    -0.000314598750f, -0.000336418978f, -0.000359734009f, -0.000384644647f,
    -0.000411258354f, -0.000439689678f, -0.000470060709f, -0.000502501557f,
    -0.000537150867f, -0.000574156355f, -0.000613675392f, -0.000655875606f,
    -0.000700935531f, -0.000749045294f, -0.000800407338f, -0.000855237192f,
    -0.000913764284f, -0.000976232803f, -0.001042902614f, -0.001114050214f,
    -0.001189969760f, -0.001270974146f, -0.001357396136f, -0.001449589573f,
    -0.001547930649f, -0.001652819240f, -0.001764680328f, -0.001883965486f,
    -0.002011154451f, -0.002146756779f, -0.002291313589f, -0.002445399391f,
    -0.002609624017f, -0.002784634645f, -0.002971117924f, -0.003169802206f,
    -0.003381459883f, -0.003606909840f, -0.003847020016f, -0.004102710089f,
    -0.004374954273f, -0.004664784247f, -0.004973292195f, -0.005301633984f,
    -0.005651032459f, -0.006022780865f, -0.006418246400f, -0.006838873881f,
    -0.007286189544f, -0.007761804944f, -0.008267420984f, -0.008804832034f,
    -0.009375930150f, -0.009982709382f, -0.010627270142f, -0.011311823633f,
    -0.012038696315f, -0.012810334376f, -0.013629308191f, -0.014498316738f,
    -0.015420191928f, -0.016397902812f, -0.017434559617f, -0.018533417555f,
    -0.019697880343f, -0.020931503363f, -0.022237996384f, -0.023621225743f,
    -0.025085215910f, -0.026634150281f, -0.028272371115f, -0.030004378428f,
    -0.031834827716f, -0.033768526301f, -0.035810428118f, -0.037965626711f,
    -0.040239346190f, -0.042636929886f, -0.045163826396f, -0.047825572693f,
    -0.050627773935f, -0.053576079592f, -0.056676155448f, -0.059933651014f,
    -0.063354161868f, -0.066943186351f, -0.070706076060f, -0.074647979518f,
    -0.078773778353f, -0.083088015310f, -0.087594813369f, -0.092297785215f,
    -0.097199932315f, -0.102303532798f, -0.107610017402f, -0.113119832706f,
    -0.118832290945f, -0.124745405760f, -0.130855713302f, -0.137158078258f,
    -0.143645484496f, -0.150308810262f, -0.157136588078f, -0.164114749820f,
    -0.171226357826f, -0.178451323291f, -0.185766113763f, -0.193143452081f,
    -0.200552009794f, -0.207956098813f, -0.215315365860f, -0.222584495122f,
    -0.229712925434f, -0.236644589255f, -0.243317681599f, -0.249664467991f,
    -0.255611141330f, -0.261077738174f, -0.265978125496f, -0.270220069141f,
    -0.273705395141f, -0.276330254541f, -0.277985501459f, -0.278557192623f,
    -0.277927214632f, -0.275974042583f, -0.272573630550f, -0.267600430710f,
    -0.260928533747f, -0.252432918694f, -0.241990795705f, -0.229483020597f,
    -0.214795555613f, -0.197820946950f, -0.178459786438f, -0.156622122604f,
    -0.132228785369f, -0.105212589022f, -0.075519379953f, -0.043108898898f,
    -0.007955432116f, 0.029951768224f,  0.070608307356f,  0.113994720397f,
    0.160076791741f,  0.208806111511f,  0.260120852100f,  0.313946741537f,
    0.370198205218f,  0.428779643703f,  0.489586811792f,  0.552508263155f,
    0.617426825163f,  0.684221070309f,  0.752766753312f,  0.822938186673f,
    0.894609531639f,  0.967655986107f,  1.041954855681f,  1.117386498687f,
    1.193835140227f,  1.271189554285f,  1.349343616257f,  1.428196731135f,
    1.507654144811f,  1.587627147660f,  1.668033180730f,  1.748795855517f,
    1.829844898577f,  1.911116032122f,  1.992550801364f,  2.074096358779f,
    2.155705214697f,  2.237334962784f,  2.318947988046f,  2.400511164082f,
    2.481995545357f,  2.563376059425f,  2.644631203185f,  2.725742746485f,
    2.806695445713f,  2.887476769407f,  2.968076637341f,  3.048487174140f,
    3.128702478008f,  3.208718404888f,  3.288532368044f,  3.368143152875f,
    3.447550746568f,  3.526756182070f,  3.605761395761f,  3.684569098138f,
    3.763182656766f,  3.841605990732f,  3.919843475836f,  3.997899859739f,
    4.075780186311f,  4.153489728464f,  4.231033928743f,  4.308418347017f,
    4.385648614645f,  4.462730394497f,  4.539669346292f,  4.616471096730f,
    4.693141213927f,  4.769685185729f,  4.846108401469f,  4.922416136819f,
    4.998613541386f,  5.074705628726f,  5.150697268515f,  5.226593180611f,
    5.302397930762f,  5.378115927776f,  5.453751421942f,  5.529308504540f,
    5.604791108293f,  5.680203008624f,  5.755547825582f,  5.830829026356f,
    5.906049928253f,  5.981213702080f,  6.056323375837f,  6.131381838663f,
    6.206391844979f,  6.281356018769f,  6.356276857966f,  6.431156738895f,
    6.505997920746f,  6.580802550041f,  6.655572665075f,  6.730310200330f,
    6.805016990792f,  6.879694776213f,  6.954345205283f,  7.028969839682f,
    7.103570158049f,  7.178147559817f,  7.252703368935f,  7.327238837474f,
    7.401755149105f,  7.476253422433f,  7.550734714247f,  7.625200022602f,
    7.699650289794f,  7.774086405219f,  7.848509208110f,  7.922919490136f,
    7.997318040654f,
};

at::Half CalcSwishByLUT(at::Half x) {
  const at::Half a = (at::Half)(-20.5);
  const at::Half b = (at::Half)(8.0);
  const int nBins = 384;

  if ((x > b) || (x == 0.0)) {
    return x;
  }

  at::Half delta = (b - a) / (at::Half)nBins;
  at::Half one_over_delta = at::Half(1) / delta;
  at::Half a_one_over_delta = a * one_over_delta;

  // Clamp the input in the range of a to b
  if (x < a) {
    x = a;
  }

  if (x > b) {
    x = b;
  }
  /*
   * bin_calc = (x - a) * one_over_delta;
   * Use MAC bin_calc = x * one_over_delta - a_one_over_delta;
   */

  float f_x = x;
  float f_one_over_delta = one_over_delta;
  float f_a_one_over_delta = -a_one_over_delta;
  fma_fp16(1, &f_x, &f_one_over_delta, &f_a_one_over_delta);
  at::Half bin_calc = f_a_one_over_delta;

  uint32_t bin = bin_calc < 0 ? 0 : (uint32_t)floor(bin_calc);
  // Clamp bin to nBins-2, to have valid LUT access i.e. b+1 = 255 (for LUT size
  // of 256)
  if (bin > (nBins - 2)) {
    bin = nBins - 2;
  }

  // Use MAC bin_x = a + delta * at::Half(bin);

  float f_delta = delta;
  float f_bin = at::Half(bin);
  float f_a = a;
  fma_fp16(1, &f_delta, &f_bin, &f_a);
  at::Half bin_x = at::Half(f_a);

  at::Half p = at::Half(x - bin_x) * one_over_delta;

  at::Half res1 = swishLutKnot[bin + 1] * p;
  float f_p = -p;
  float lutVal = at::Half(swishLutKnot[bin]);

  fma_fp16(1, &f_p, &lutVal, &lutVal);
  at::Half res2 = lutVal;

  return at::Half(res1 + res2);
}
static const float swishLutKnotCub[] = {
    -0.00000000e+00f, -0.00000000e+00f, -0.00000000e+00f, -0.00000000e+00f,
    -0.00000000e+00f, -0.00000000e+00f, -0.00000000e+00f, -0.00000000e+00f,
    -0.00000000e+00f, -5.96046448e-08f, -5.96046448e-08f, -5.96046448e-08f,
    -5.96046448e-08f, -5.96046448e-08f, -5.96046448e-08f, -5.96046448e-08f,
    -5.96046448e-08f, -5.96046448e-08f, -1.19209290e-07f, -1.19209290e-07f,
    -1.19209290e-07f, -1.19209290e-07f, -1.78813934e-07f, -1.78813934e-07f,
    -1.78813934e-07f, -2.38418579e-07f, -2.38418579e-07f, -2.98023224e-07f,
    -2.98023224e-07f, -3.57627869e-07f, -4.17232513e-07f, -4.17232513e-07f,
    -4.76837158e-07f, -5.36441803e-07f, -6.55651093e-07f, -7.15255737e-07f,
    -7.74860382e-07f, -8.94069672e-07f, -1.01327896e-06f, -1.13248825e-06f,
    -1.25169754e-06f, -1.43051147e-06f, -1.60932541e-06f, -1.78813934e-06f,
    -2.02655792e-06f, -2.26497650e-06f, -2.56299973e-06f, -2.92062759e-06f,
    -3.27825546e-06f, -3.63588333e-06f, -4.11272049e-06f, -4.58955765e-06f,
    -5.18560410e-06f, -5.84125519e-06f, -6.55651093e-06f, -7.33137131e-06f,
    -8.22544098e-06f, -9.23871994e-06f, -1.03712082e-05f, -1.16825104e-05f,
    -1.31130219e-05f, -1.46627426e-05f, -1.65104866e-05f, -1.84774399e-05f,
    -2.07424164e-05f, -2.33054161e-05f, -2.61068344e-05f, -2.93254852e-05f,
    -3.29017639e-05f, -3.68356705e-05f, -4.13656235e-05f, -4.63724136e-05f,
    -5.19752502e-05f, -5.82337379e-05f, -6.52670860e-05f, -7.31945038e-05f,
    -8.19563866e-05f, -9.18507576e-05f, -1.02877617e-04f, -1.15275383e-04f,
    -1.29103661e-04f, -1.44600868e-04f, -1.61886215e-04f, -1.81198120e-04f,
    -2.02775002e-04f, -2.26974487e-04f, -2.53915787e-04f, -2.84194946e-04f,
    -3.17811966e-04f, -3.55482101e-04f, -3.97443771e-04f, -4.44412231e-04f,
    -4.96864319e-04f, -5.55038452e-04f, -6.20365143e-04f, -6.93321228e-04f,
    -7.74383545e-04f, -8.64505768e-04f, -9.65118408e-04f, -1.07765198e-03f,
    -1.20258331e-03f, -1.34181976e-03f, -1.49631500e-03f, -1.66797638e-03f,
    -1.85966492e-03f, -2.07328796e-03f, -2.30979919e-03f, -2.57301331e-03f,
    -2.86483765e-03f, -3.18908691e-03f, -3.54766846e-03f, -3.94821167e-03f,
    -4.39071655e-03f, -4.87899780e-03f, -5.42068481e-03f, -6.01959229e-03f,
    -6.68334961e-03f, -7.41958618e-03f, -8.22448730e-03f, -9.12475586e-03f,
    -1.01089478e-02f, -1.11923218e-02f, -1.23901367e-02f, -1.37023926e-02f,
    -1.51443481e-02f, -1.67388916e-02f, -1.84631348e-02f, -2.03704834e-02f,
    -2.24456787e-02f, -2.47192383e-02f, -2.71911621e-02f, -2.98919678e-02f,
    -3.28063965e-02f, -3.59802246e-02f, -3.93981934e-02f, -4.30908203e-02f,
    -4.70581055e-02f, -5.13000488e-02f, -5.58471680e-02f, -6.06689453e-02f,
    -6.57348633e-02f, -7.11669922e-02f, -7.67822266e-02f, -8.26416016e-02f,
    -8.86840820e-02f, -9.48486328e-02f, -1.01074219e-01f, -1.07238770e-01f,
    -1.13342285e-01f, -1.19201660e-01f, -1.24633789e-01f, -1.29516602e-01f,
    -1.33666992e-01f, -1.36840820e-01f, -1.38793945e-01f, -1.39160156e-01f,
    -1.37817383e-01f, -1.34521484e-01f, -1.28662109e-01f, -1.20300293e-01f,
    -1.08947754e-01f, -9.43603516e-02f, -7.63549805e-02f, -5.47180176e-02f,
    -2.92968750e-02f, 0.00000000e+00f,  3.32031250e-02f,  7.02514648e-02f,
    1.11145020e-01f,  1.55639648e-01f,  2.03491211e-01f,  2.54638672e-01f,
    3.08837891e-01f,  3.65478516e-01f,  4.24560547e-01f,  4.85839844e-01f,
    5.48828125e-01f,  6.13281250e-01f,  6.78710938e-01f,  7.45605469e-01f,
    8.12988281e-01f,  8.80859375e-01f,  9.49218750e-01f,  1.01757812e+00f,
    1.08691406e+00f,  1.15527344e+00f,  1.22363281e+00f,  1.29199219e+00f,
    1.36035156e+00f,  1.42871094e+00f,  1.49707031e+00f,  1.56445312e+00f,
    1.63183594e+00f,  1.69824219e+00f,  1.76562500e+00f,  1.83203125e+00f,
    1.89843750e+00f,  1.96386719e+00f,  2.02929688e+00f,  2.09570312e+00f,
    2.16015625e+00f,  2.22460938e+00f,  2.29101562e+00f,  2.35546875e+00f,
    2.41992188e+00f,  2.48242188e+00f,  2.54687500e+00f,  2.61132812e+00f,
    2.67578125e+00f,  2.73828125e+00f,  2.80273438e+00f,  2.86523438e+00f,
    2.92968750e+00f,  2.99218750e+00f,  3.05664062e+00f,  3.11914062e+00f,
    3.18164062e+00f,  3.24609375e+00f,  3.30859375e+00f,  3.37109375e+00f,
    3.43359375e+00f,  3.49609375e+00f,  3.56054688e+00f,  3.62304688e+00f,
    3.68554688e+00f,  3.74804688e+00f,  3.81054688e+00f,  3.87304688e+00f,
    3.93554688e+00f,  3.99804688e+00f,  4.06250000e+00f};

at::Half CalcSwishByLUTCubic(at::Half x) {
  const float SWISH_KNOT_RANGE_MIN = -20.5f;
  const float SWISH_KNOT_RANGE_MAX = 8.0f;
  const float SWISH_KNOT_LUT_DELTA = 0.125000f;
  const int SWISH_KNOT_LUT_BIAS = 165;

  at::Half x_min = (at::Half)SWISH_KNOT_RANGE_MIN;
  at::Half x_max = (at::Half)SWISH_KNOT_RANGE_MAX;
  at::Half delta = (at::Half)SWISH_KNOT_LUT_DELTA;
  at::Half bias = SWISH_KNOT_LUT_BIAS;

  if (x > SWISH_KNOT_RANGE_MAX) {
    return x;
  }

  at::Half one_over_delta = at::Half(1) / delta;

  // Clamp the input in the range of a to b
  if (x < x_min) {
    x = x_min;
  }
  if (x > x_max) {
    x = x_max;
  }
  at::Half x_over_delta = x * one_over_delta;
  at::Half x_over_delta_int = std::round(x_over_delta);
  at::Half p = x_over_delta - x_over_delta_int;

  x_over_delta_int = x_over_delta_int + bias;
  uint32_t k_bin = (uint32_t)x_over_delta_int;

  at::Half y_left = swishLutKnotCub[k_bin - 1];
  at::Half y_mid = swishLutKnotCub[k_bin];
  at::Half y_right = swishLutKnotCub[k_bin + 1];

  at::Half a = y_mid + y_mid;
  at::Half c = (y_right + y_left);
  c = c - a;
  at::Half b = y_right - y_left;

  at::Half result = std::fma(p, c, b);
  result = result * p;
  result = result + a;

  if (x == (at::Half)0.0f) {
    result = x;
  }
  return result;
}

at::Half CalcLogit(at::Half input, float eps) {
  // Clamp the input in the range of eps to (1-eps)
  float x = at::Half(input);
  if (at::Half(input) < at::Half(eps)) {
    x = at::Half(eps);
  }
  if (at::Half(input) > at::Half(1 - eps)) {
    x = at::Half(1 - eps);
  }
  if (x < 0.0f || x > 1.0f) {
    return at::Half(NAN);
  } else {
    if (x < eps) {
      float lower_bound = log(eps / (1.0 - eps));
      return at::Half(lower_bound);
    } else if (input >= (1.0f - eps)) {
      float upper_bound = log((1.0 - eps) / eps);
      return at::Half(upper_bound);
    } else {
      return at::Half(log((x / (1 - x))));
    }
  }
}

} // namespace fake_fp16

namespace caffe2 {
using namespace fake_fp16;

struct SigmoidEmulatorFunctor {
  bool operator()(
      const int N,
      const float* X,
      float* Y,
      CPUContext* /* unused */) const {
    for (int i = 0; i < N; i++) {
      Y[i] = CalcSigmoidByLUT((at::Half)X[i]);
    }
    return true;
  }
};

struct TanhEmulatorFunctor {
  bool operator()(
      const int N,
      const float* X,
      float* Y,
      CPUContext* /* unused */) const {
    for (int i = 0; i < N; i++) {
      Y[i] = CalcTanhByLUT((at::Half)X[i]);
    }
    return true;
  }
};

OpSchema::Cost CostInferenceForRelu(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<0>(def, in);
  cost.params_bytes = 0;
  return cost;
}

REGISTER_CPU_OPERATOR(
    ReluFakeFp16,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReluFakeFp16Functor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(ReluFakeFp16)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForRelu)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Applies rectified linear unit operation to the input data element-wise. The Relu operation takes one input $X$, produces one output $Y$, and is defined as:

$$Y = max(0,X)$$

The input of this operator is converted to fp16 precision. And since the ReLU
op doesn't have any arithmetics, there is no need to convert the output.

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
  "ReluFakeFp16",
  ["X"],
  ["Y"]
  )

workspace.FeedBlob("X", np.random.randn(4, 4).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

X:
 [[-1.4655551   0.64575136  0.7921748   0.4150579 ]
 [ 0.41085166 -0.2837964   0.9881425  -1.9300346 ]
 [ 0.39705405  0.44639114  0.9940703   0.2926532 ]
 [-0.6726489   0.01330667  1.101319    0.33858967]]

Y:
 [[0.         0.64575136 0.7921748  0.4150579 ]
 [0.41085166 0.         0.9881425  0.        ]
 [0.39705405 0.44639114 0.9940703  0.2926532 ]
 [0.         0.01330667 1.101319   0.33858967]]

```

</details>


)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor with same shape as input")
    .InheritOnnxSchema();

REGISTER_CPU_OPERATOR(
    SigmoidFakeFp16NNPI,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SigmoidEmulatorFunctor>);
OPERATOR_SCHEMA(SigmoidFakeFp16NNPI).NumInputs(1).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    SigmoidFakeFp16,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SigmoidFakeIdealFp16Functor>);

// Input: X, output: Y
OPERATOR_SCHEMA(SigmoidFakeFp16)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Apply the Sigmoid function element-wise to the input tensor. This is often used
as a non-linear activation function in a neural network. The sigmoid function is
defined as:

$$Sigmoid(x) = \frac{1}{1+\exp(-x)}$$

The input and output of this operator are converted to fp16 precision.

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "SigmoidFakeFp16",
    ["X"],
    ["Y"]
)

workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
print("input:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("sigmoid:", workspace.FetchBlob("Y"))

```

**Result**

```

input: [ 1.5744036   0.31632107  1.7842269   1.4450722  -2.1726978 ]
sigmoid: [0.8284105  0.57842743 0.85621804 0.80923885 0.10222916]

```

</details>


)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.")
    .InheritOnnxSchema();

REGISTER_CPU_OPERATOR(
    SqrFakeFp16,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SqrFakeFp16Functor<CPUContext>>);

OPERATOR_SCHEMA(SqrFakeFp16)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Performs element-wise squaring ($x^2$) of input tensor. Inputs are converted
to fp16 before the operation, computation is in fp32, and the result is
also converted to fp16.


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "SqrFakeFp16",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[4. 6. 2.]
 [0. 1. 6.]
 [9. 2. 7.]]
Y:
[[16. 36.  4.]
 [ 0.  1. 36.]
 [81.  4. 49.]]

```

</details>

    )DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input data tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.");

REGISTER_CPU_OPERATOR(
    TanhFakeFp16NNPI,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, TanhEmulatorFunctor>);
OPERATOR_SCHEMA(TanhFakeFp16NNPI).NumInputs(1).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    TanhFakeFp16,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        TanhFakeIdealFp16Functor>);
OPERATOR_SCHEMA(TanhFakeFp16)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.

The input and output of this operator are converted to fp16 precision.

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "TanhFakeFp16",
    ["X"],
    ["X"],
)

workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("X:\n", workspace.FetchBlob("X"))

```

**Result**

```

X:
 [[ 2.032603   -2.3556721  -0.14955314]
 [ 0.39309832 -1.1020128  -0.92951244]
 [-0.62815386  0.21342885  1.4002231 ]]

X:
 [[ 0.9662601  -0.982175   -0.14844811]
 [ 0.3740282  -0.8012209  -0.73036647]
 [-0.55677974  0.21024609  0.8853999 ]]

```

</details>

)DOC")
    .Input(0, "input", "1-D input tensor")
    .Output(
        0,
        "output",
        "The hyperbolic tangent values of the input tensor, computed "
        "element-wise")
    .InheritOnnxSchema();

struct SwishEmulatorFunctor {
  bool operator()(
      const int N,
      const float* X,
      float* Y,
      CPUContext* /* unused */) const {
    for (int i = 0; i < N; i++) {
      Y[i] = CalcSwishByLUT((at::Half)X[i]);
    }
    return true;
  }
};

template <class Context>
class LogitEmulatorFunctor final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit LogitEmulatorFunctor(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(float, "eps", eps_, 1e-6f) {}
  ~LogitEmulatorFunctor() noexcept override {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const int N = X.numel();
    auto* Y = Output(0, X.sizes(), at::dtype<float>());
    Y->ResizeLike(X);
    const float* X_data = X.template data<float>();
    float* Y_data = Y->template mutable_data<float>();
    std::vector<float> X_rounded(N);
    fbgemm::RoundToFloat16(
        X_data, X_rounded.data(), N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    X_data = X_rounded.data();
    for (int i = 0; i < N; i++) {
      Y_data[i] = CalcLogit((at::Half)X_data[i], eps_);
    }
    return true;
  }

 private:
  const float eps_;
};

REGISTER_CPU_OPERATOR(
    SwishFakeFp16NNPI,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SwishEmulatorFunctor>);

// Input: X, output: Y
OPERATOR_SCHEMA(SwishFakeFp16NNPI)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Apply the Swish function element-wise to the input tensor.

$$Swish(x) = \frac{x}{1+\exp(-x)}$$

The input and output of this operator are converted to fp16 precision.

<details>
</details>


)DOC")
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(0, "Y", "*(type: Tensor`<float>`)* Output tensor.")
    .InheritOnnxSchema();

REGISTER_CPU_OPERATOR(LogitFakeFp16NNPI, LogitEmulatorFunctor<CPUContext>);

OPERATOR_SCHEMA(LogitFakeFp16NNPI)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
      Elementwise logit fake fp16 transform:
      $$logit(x) = log(\frac{x}{(1 - x)})$$
      where x is the input data clampped in (eps, 1-eps).)DOC")
    .Arg("eps (optional)", "small positive epsilon value, the default is 1e-6.")
    .Input(0, "X", "input float tensor")
    .Output(0, "Y", "output float tensor");

} // namespace caffe2

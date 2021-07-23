#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <c10/util/complex.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

template <> class Vectorized<c10::complex<float>> {
private:
  __m512 values;
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
public:
  using value_type = c10::complex<float>;
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }
  Vectorized() {}
  Vectorized(__m512 v) : values(v) {}
  Vectorized(c10::complex<float> val) {
    float real_value = val.real();
    float imag_value = val.imag();
    values = _mm512_setr_ps(real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value);
  }
  Vectorized(c10::complex<float> val1, c10::complex<float> val2,
            c10::complex<float> val3, c10::complex<float> val4,
            c10::complex<float> val5, c10::complex<float> val6,
            c10::complex<float> val7, c10::complex<float> val8) {
    values = _mm512_setr_ps(val1.real(), val1.imag(),
                            val2.real(), val2.imag(),
                            val3.real(), val3.imag(),
                            val4.real(), val4.imag(),
                            val5.real(), val5.imag(),
                            val6.real(), val6.imag(),
                            val7.real(), val7.imag(),
                            val8.real(), val8.imag());
  }
  operator __m512() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<c10::complex<float>> blend(const Vectorized<c10::complex<float>>& a,
                                              const Vectorized<c10::complex<float>>& b) {
    // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    // NOLINTNEXTLINE(clang-diagnostic-warning)
    // The compiler would hopefully convert this switch condition
    // into a jump table
    switch (mask) {
      case 0:
        return a;
      case 1:
        return _mm512_mask_blend_ps(0x03, a.values, b.values);
      case 2:
        return _mm512_mask_blend_ps(0x0C, a.values, b.values);
      case 3:
        return _mm512_mask_blend_ps(0x0F, a.values, b.values);
      case 4:
        return _mm512_mask_blend_ps(0x30, a.values, b.values);
      case 5:
        return _mm512_mask_blend_ps(0x33, a.values, b.values);
      case 6:
        return _mm512_mask_blend_ps(0x3C, a.values, b.values);
      case 7:
        return _mm512_mask_blend_ps(0x3F, a.values, b.values);
      case 8:
        return _mm512_mask_blend_ps(0xC0, a.values, b.values);
      case 9:
        return _mm512_mask_blend_ps(0xC3, a.values, b.values);
      case 10:
        return _mm512_mask_blend_ps(0xCC, a.values, b.values);
      case 11:
        return _mm512_mask_blend_ps(0xCF, a.values, b.values);
      case 12:
        return _mm512_mask_blend_ps(0xF0, a.values, b.values);
      case 13:
        return _mm512_mask_blend_ps(0xF3, a.values, b.values);
      case 14:
        return _mm512_mask_blend_ps(0xFC, a.values, b.values);
      case 15:
        return _mm512_mask_blend_ps(0xFF, a.values, b.values);
      case 16:
        return _mm512_mask_blend_ps(0x300, a.values, b.values);
      case 17:
        return _mm512_mask_blend_ps(0x303, a.values, b.values);
      case 18:
        return _mm512_mask_blend_ps(0x30C, a.values, b.values);
      case 19:
        return _mm512_mask_blend_ps(0x30F, a.values, b.values);
      case 20:
        return _mm512_mask_blend_ps(0x330, a.values, b.values);
      case 21:
        return _mm512_mask_blend_ps(0x333, a.values, b.values);
      case 22:
        return _mm512_mask_blend_ps(0x33C, a.values, b.values);
      case 23:
        return _mm512_mask_blend_ps(0x33F, a.values, b.values);
      case 24:
        return _mm512_mask_blend_ps(0x3C0, a.values, b.values);
      case 25:
        return _mm512_mask_blend_ps(0x3C3, a.values, b.values);
      case 26:
        return _mm512_mask_blend_ps(0x3CC, a.values, b.values);
      case 27:
        return _mm512_mask_blend_ps(0x3CF, a.values, b.values);
      case 28:
        return _mm512_mask_blend_ps(0x3F0, a.values, b.values);
      case 29:
        return _mm512_mask_blend_ps(0x3F3, a.values, b.values);
      case 30:
        return _mm512_mask_blend_ps(0x3FC, a.values, b.values);
      case 31:
        return _mm512_mask_blend_ps(0x3FF, a.values, b.values);
      case 32:
        return _mm512_mask_blend_ps(0xC00, a.values, b.values);
      case 33:
        return _mm512_mask_blend_ps(0xC03, a.values, b.values);
      case 34:
        return _mm512_mask_blend_ps(0xC0C, a.values, b.values);
      case 35:
        return _mm512_mask_blend_ps(0xC0F, a.values, b.values);
      case 36:
        return _mm512_mask_blend_ps(0xC30, a.values, b.values);
      case 37:
        return _mm512_mask_blend_ps(0xC33, a.values, b.values);
      case 38:
        return _mm512_mask_blend_ps(0xC3C, a.values, b.values);
      case 39:
        return _mm512_mask_blend_ps(0xC3F, a.values, b.values);
      case 40:
        return _mm512_mask_blend_ps(0xCC0, a.values, b.values);
      case 41:
        return _mm512_mask_blend_ps(0xCC3, a.values, b.values);
      case 42:
        return _mm512_mask_blend_ps(0xCCC, a.values, b.values);
      case 43:
        return _mm512_mask_blend_ps(0xCCF, a.values, b.values);
      case 44:
        return _mm512_mask_blend_ps(0xCF0, a.values, b.values);
      case 45:
        return _mm512_mask_blend_ps(0xCF3, a.values, b.values);
      case 46:
        return _mm512_mask_blend_ps(0xCFC, a.values, b.values);
      case 47:
        return _mm512_mask_blend_ps(0xCFF, a.values, b.values);
      case 48:
        return _mm512_mask_blend_ps(0xF00, a.values, b.values);
      case 49:
        return _mm512_mask_blend_ps(0xF03, a.values, b.values);
      case 50:
        return _mm512_mask_blend_ps(0xF0C, a.values, b.values);
      case 51:
        return _mm512_mask_blend_ps(0xF0F, a.values, b.values);
      case 52:
        return _mm512_mask_blend_ps(0xF30, a.values, b.values);
      case 53:
        return _mm512_mask_blend_ps(0xF33, a.values, b.values);
      case 54:
        return _mm512_mask_blend_ps(0xF3C, a.values, b.values);
      case 55:
        return _mm512_mask_blend_ps(0xF3F, a.values, b.values);
      case 56:
        return _mm512_mask_blend_ps(0xFC0, a.values, b.values);
      case 57:
        return _mm512_mask_blend_ps(0xFC3, a.values, b.values);
      case 58:
        return _mm512_mask_blend_ps(0xFCC, a.values, b.values);
      case 59:
        return _mm512_mask_blend_ps(0xFCF, a.values, b.values);
      case 60:
        return _mm512_mask_blend_ps(0xFF0, a.values, b.values);
      case 61:
        return _mm512_mask_blend_ps(0xFF3, a.values, b.values);
      case 62:
        return _mm512_mask_blend_ps(0xFFC, a.values, b.values);
      case 63:
        return _mm512_mask_blend_ps(0xFFF, a.values, b.values);
      case 64:
        return _mm512_mask_blend_ps(0x3000, a.values, b.values);
      case 65:
        return _mm512_mask_blend_ps(0x3003, a.values, b.values);
      case 66:
        return _mm512_mask_blend_ps(0x300C, a.values, b.values);
      case 67:
        return _mm512_mask_blend_ps(0x300F, a.values, b.values);
      case 68:
        return _mm512_mask_blend_ps(0x3030, a.values, b.values);
      case 69:
        return _mm512_mask_blend_ps(0x3033, a.values, b.values);
      case 70:
        return _mm512_mask_blend_ps(0x303C, a.values, b.values);
      case 71:
        return _mm512_mask_blend_ps(0x303F, a.values, b.values);
      case 72:
        return _mm512_mask_blend_ps(0x30C0, a.values, b.values);
      case 73:
        return _mm512_mask_blend_ps(0X30C3, a.values, b.values);
      case 74:
        return _mm512_mask_blend_ps(0x30CC, a.values, b.values);
      case 75:
        return _mm512_mask_blend_ps(0x30CF, a.values, b.values);
      case 76:
        return _mm512_mask_blend_ps(0x30F0, a.values, b.values);
      case 77:
        return _mm512_mask_blend_ps(0x30F3, a.values, b.values);
      case 78:
        return _mm512_mask_blend_ps(0x30FC, a.values, b.values);
      case 79:
        return _mm512_mask_blend_ps(0x30FF, a.values, b.values);
      case 80:
        return _mm512_mask_blend_ps(0x3300, a.values, b.values);
      case 81:
        return _mm512_mask_blend_ps(0X3303, a.values, b.values);
      case 82:
        return _mm512_mask_blend_ps(0x330C, a.values, b.values);
      case 83:
        return _mm512_mask_blend_ps(0x330F, a.values, b.values);
      case 84:
        return _mm512_mask_blend_ps(0x3330, a.values, b.values);
      case 85:
        return _mm512_mask_blend_ps(0x3333, a.values, b.values);
      case 86:
        return _mm512_mask_blend_ps(0x333C, a.values, b.values);
      case 87:
        return _mm512_mask_blend_ps(0X333F, a.values, b.values);
      case 88:
        return _mm512_mask_blend_ps(0x33C0, a.values, b.values);
      case 89:
        return _mm512_mask_blend_ps(0x33C3, a.values, b.values);
      case 90:
        return _mm512_mask_blend_ps(0x33CC, a.values, b.values);
      case 91:
        return _mm512_mask_blend_ps(0x33CF, a.values, b.values);
      case 92:
        return _mm512_mask_blend_ps(0x33F0, a.values, b.values);
      case 93:
        return _mm512_mask_blend_ps(0x33F3, a.values, b.values);
      case 94:
        return _mm512_mask_blend_ps(0x33FC, a.values, b.values);
      case 95:
        return _mm512_mask_blend_ps(0x33FF, a.values, b.values);
      case 96:
        return _mm512_mask_blend_ps(0X3C00, a.values, b.values);
      case 97:
        return _mm512_mask_blend_ps(0x3C03, a.values, b.values);
      case 98:
        return _mm512_mask_blend_ps(0x3C0C, a.values, b.values);
      case 99:
        return _mm512_mask_blend_ps(0x3C0F, a.values, b.values);
      case 100:
        return _mm512_mask_blend_ps(0x3C30, a.values, b.values);
      case 101:
        return _mm512_mask_blend_ps(0x3C33, a.values, b.values);
      case 102:
        return _mm512_mask_blend_ps(0x3C3C, a.values, b.values);
      case 103:
        return _mm512_mask_blend_ps(0x3C3F, a.values, b.values);
      case 104:
        return _mm512_mask_blend_ps(0x3CC0, a.values, b.values);
      case 105:
        return _mm512_mask_blend_ps(0x3CC3, a.values, b.values);
      case 106:
        return _mm512_mask_blend_ps(0x3CCC, a.values, b.values);
      case 107:
        return _mm512_mask_blend_ps(0x3CCF, a.values, b.values);
      case 108:
        return _mm512_mask_blend_ps(0x3CF0, a.values, b.values);
      case 109:
        return _mm512_mask_blend_ps(0x3CF3, a.values, b.values);
      case 110:
        return _mm512_mask_blend_ps(0x3CFC, a.values, b.values);
      case 111:
        return _mm512_mask_blend_ps(0x3CFF, a.values, b.values);
      case 112:
        return _mm512_mask_blend_ps(0x3F00, a.values, b.values);
      case 113:
        return _mm512_mask_blend_ps(0x3F03, a.values, b.values);
      case 114:
        return _mm512_mask_blend_ps(0x3F0C, a.values, b.values);
      case 115:
        return _mm512_mask_blend_ps(0x3F0F, a.values, b.values);
      case 116:
        return _mm512_mask_blend_ps(0x3F30, a.values, b.values);
      case 117:
        return _mm512_mask_blend_ps(0x3F33, a.values, b.values);
      case 118:
        return _mm512_mask_blend_ps(0x3F3C, a.values, b.values);
      case 119:
        return _mm512_mask_blend_ps(0x3F3F, a.values, b.values);
      case 120:
        return _mm512_mask_blend_ps(0x3FC0, a.values, b.values);
      case 121:
        return _mm512_mask_blend_ps(0x3FC3, a.values, b.values);
      case 122:
        return _mm512_mask_blend_ps(0x3FCC, a.values, b.values);
      case 123:
        return _mm512_mask_blend_ps(0x3FCF, a.values, b.values);
      case 124:
        return _mm512_mask_blend_ps(0x3FF0, a.values, b.values);
      case 125:
        return _mm512_mask_blend_ps(0x3FF3, a.values, b.values);
      case 126:
        return _mm512_mask_blend_ps(0x3FFC, a.values, b.values);
      case 127:
        return _mm512_mask_blend_ps(0x3FFF, a.values, b.values);
      case 128:
        return _mm512_mask_blend_ps(0xC000, a.values, b.values);
      case 129:
        return _mm512_mask_blend_ps(0xC003, a.values, b.values);
      case 130:
        return _mm512_mask_blend_ps(0xC00C, a.values, b.values);
      case 131:
        return _mm512_mask_blend_ps(0xC00F, a.values, b.values);
      case 132:
        return _mm512_mask_blend_ps(0xC030, a.values, b.values);
      case 133:
        return _mm512_mask_blend_ps(0xC033, a.values, b.values);
      case 134:
        return _mm512_mask_blend_ps(0xC03C, a.values, b.values);
      case 135:
        return _mm512_mask_blend_ps(0xC03F, a.values, b.values);
      case 136:
        return _mm512_mask_blend_ps(0xC0C0, a.values, b.values);
      case 137:
        return _mm512_mask_blend_ps(0xC0C3, a.values, b.values);
      case 138:
        return _mm512_mask_blend_ps(0xC0CC, a.values, b.values);
      case 139:
        return _mm512_mask_blend_ps(0xC0CF, a.values, b.values);
      case 140:
        return _mm512_mask_blend_ps(0xC0F0, a.values, b.values);
      case 141:
        return _mm512_mask_blend_ps(0xC0F3, a.values, b.values);
      case 142:
        return _mm512_mask_blend_ps(0xC0FC, a.values, b.values);
      case 143:
        return _mm512_mask_blend_ps(0xC0FF, a.values, b.values);
      case 144:
        return _mm512_mask_blend_ps(0xC300, a.values, b.values);
      case 145:
        return _mm512_mask_blend_ps(0xC303, a.values, b.values);
      case 146:
        return _mm512_mask_blend_ps(0xC30C, a.values, b.values);
      case 147:
        return _mm512_mask_blend_ps(0xC30F, a.values, b.values);
      case 148:
        return _mm512_mask_blend_ps(0xC330, a.values, b.values);
      case 149:
        return _mm512_mask_blend_ps(0xC333, a.values, b.values);
      case 150:
        return _mm512_mask_blend_ps(0xC33C, a.values, b.values);
      case 151:
        return _mm512_mask_blend_ps(0xC33F, a.values, b.values);
      case 152:
        return _mm512_mask_blend_ps(0xC3C0, a.values, b.values);
      case 153:
        return _mm512_mask_blend_ps(0xC3C3, a.values, b.values);
      case 154:
        return _mm512_mask_blend_ps(0xC3CC, a.values, b.values);
      case 155:
        return _mm512_mask_blend_ps(0xC3CF, a.values, b.values);
      case 156:
        return _mm512_mask_blend_ps(0xC3F0, a.values, b.values);
      case 157:
        return _mm512_mask_blend_ps(0xC3F3, a.values, b.values);
      case 158:
        return _mm512_mask_blend_ps(0xC3FC, a.values, b.values);
      case 159:
        return _mm512_mask_blend_ps(0xC3FF, a.values, b.values);
      case 160:
        return _mm512_mask_blend_ps(0xCC00, a.values, b.values);
      case 161:
        return _mm512_mask_blend_ps(0xCC03, a.values, b.values);
      case 162:
        return _mm512_mask_blend_ps(0xCC0C, a.values, b.values);
      case 163:
        return _mm512_mask_blend_ps(0xCC0F, a.values, b.values);
      case 164:
        return _mm512_mask_blend_ps(0xCC30, a.values, b.values);
      case 165:
        return _mm512_mask_blend_ps(0xCC33, a.values, b.values);
      case 166:
        return _mm512_mask_blend_ps(0xCC3C, a.values, b.values);
      case 167:
        return _mm512_mask_blend_ps(0xCC3F, a.values, b.values);
      case 168:
        return _mm512_mask_blend_ps(0xCCC0, a.values, b.values);
      case 169:
        return _mm512_mask_blend_ps(0xCCC3, a.values, b.values);
      case 170:
        return _mm512_mask_blend_ps(0xCCCC, a.values, b.values);
      case 171:
        return _mm512_mask_blend_ps(0xCCCF, a.values, b.values);
      case 172:
        return _mm512_mask_blend_ps(0xCCF0, a.values, b.values);
      case 173:
        return _mm512_mask_blend_ps(0xCCF3, a.values, b.values);
      case 174:
        return _mm512_mask_blend_ps(0xCCFC, a.values, b.values);
      case 175:
        return _mm512_mask_blend_ps(0xCCFF, a.values, b.values);
      case 176:
        return _mm512_mask_blend_ps(0xCF00, a.values, b.values);
      case 177:
        return _mm512_mask_blend_ps(0xCF03, a.values, b.values);
      case 178:
        return _mm512_mask_blend_ps(0xCF0C, a.values, b.values);
      case 179:
        return _mm512_mask_blend_ps(0xCF0F, a.values, b.values);
      case 180:
        return _mm512_mask_blend_ps(0xCF30, a.values, b.values);
      case 181:
        return _mm512_mask_blend_ps(0xCF33, a.values, b.values);
      case 182:
        return _mm512_mask_blend_ps(0xCF3C, a.values, b.values);
      case 183:
        return _mm512_mask_blend_ps(0xCF3F, a.values, b.values);
      case 184:
        return _mm512_mask_blend_ps(0xCFC0, a.values, b.values);
      case 185:
        return _mm512_mask_blend_ps(0xCFC3, a.values, b.values);
      case 186:
        return _mm512_mask_blend_ps(0xCFCC, a.values, b.values);
      case 187:
        return _mm512_mask_blend_ps(0xCFCF, a.values, b.values);
      case 188:
        return _mm512_mask_blend_ps(0xCFF0, a.values, b.values);
      case 189:
        return _mm512_mask_blend_ps(0xCFF3, a.values, b.values);
      case 190:
        return _mm512_mask_blend_ps(0xCFFC, a.values, b.values);
      case 191:
        return _mm512_mask_blend_ps(0xCFFF, a.values, b.values);
      case 192:
        return _mm512_mask_blend_ps(0xF000, a.values, b.values);
      case 193:
        return _mm512_mask_blend_ps(0xF003, a.values, b.values);
      case 194:
        return _mm512_mask_blend_ps(0xF00C, a.values, b.values);
      case 195:
        return _mm512_mask_blend_ps(0xF00F, a.values, b.values);
      case 196:
        return _mm512_mask_blend_ps(0xF030, a.values, b.values);
      case 197:
        return _mm512_mask_blend_ps(0xF033, a.values, b.values);
      case 198:
        return _mm512_mask_blend_ps(0xF03C, a.values, b.values);
      case 199:
        return _mm512_mask_blend_ps(0xF03F, a.values, b.values);
      case 200:
        return _mm512_mask_blend_ps(0XF0C0, a.values, b.values);
      case 201:
        return _mm512_mask_blend_ps(0xF0C3, a.values, b.values);
      case 202:
        return _mm512_mask_blend_ps(0xF0CC, a.values, b.values);
      case 203:
        return _mm512_mask_blend_ps(0xF0CF, a.values, b.values);
      case 204:
        return _mm512_mask_blend_ps(0xF0F0, a.values, b.values);
      case 205:
        return _mm512_mask_blend_ps(0xF0F3, a.values, b.values);
      case 206:
        return _mm512_mask_blend_ps(0xF0FC, a.values, b.values);
      case 207:
        return _mm512_mask_blend_ps(0xF0FF, a.values, b.values);
      case 208:
        return _mm512_mask_blend_ps(0XF300, a.values, b.values);
      case 209:
        return _mm512_mask_blend_ps(0xF303, a.values, b.values);
      case 210:
        return _mm512_mask_blend_ps(0xF30C, a.values, b.values);
      case 211:
        return _mm512_mask_blend_ps(0xF30F, a.values, b.values);
      case 212:
        return _mm512_mask_blend_ps(0xF330, a.values, b.values);
      case 213:
        return _mm512_mask_blend_ps(0xF333, a.values, b.values);
      case 214:
        return _mm512_mask_blend_ps(0XF33C, a.values, b.values);
      case 215:
        return _mm512_mask_blend_ps(0xF33F, a.values, b.values);
      case 216:
        return _mm512_mask_blend_ps(0xF3C0, a.values, b.values);
      case 217:
        return _mm512_mask_blend_ps(0xF3C3, a.values, b.values);
      case 218:
        return _mm512_mask_blend_ps(0xF3CC, a.values, b.values);
      case 219:
        return _mm512_mask_blend_ps(0xF3CF, a.values, b.values);
      case 220:
        return _mm512_mask_blend_ps(0xF3F0, a.values, b.values);
      case 221:
        return _mm512_mask_blend_ps(0xF3F3, a.values, b.values);
      case 222:
        return _mm512_mask_blend_ps(0xF3FC, a.values, b.values);
      case 223:
        return _mm512_mask_blend_ps(0XF3FF, a.values, b.values);
      case 224:
        return _mm512_mask_blend_ps(0xFC00, a.values, b.values);
      case 225:
        return _mm512_mask_blend_ps(0xFC03, a.values, b.values);
      case 226:
        return _mm512_mask_blend_ps(0xFC0C, a.values, b.values);
      case 227:
        return _mm512_mask_blend_ps(0xFC0F, a.values, b.values);
      case 228:
        return _mm512_mask_blend_ps(0xFC30, a.values, b.values);
      case 229:
        return _mm512_mask_blend_ps(0xFC33, a.values, b.values);
      case 230:
        return _mm512_mask_blend_ps(0xFC3C, a.values, b.values);
      case 231:
        return _mm512_mask_blend_ps(0xFC3F, a.values, b.values);
      case 232:
        return _mm512_mask_blend_ps(0xFCC0, a.values, b.values);
      case 233:
        return _mm512_mask_blend_ps(0xFCC3, a.values, b.values);
      case 234:
        return _mm512_mask_blend_ps(0xFCCC, a.values, b.values);
      case 235:
        return _mm512_mask_blend_ps(0xFCCF, a.values, b.values);
      case 236:
        return _mm512_mask_blend_ps(0xFCF0, a.values, b.values);
      case 237:
        return _mm512_mask_blend_ps(0xFCF3, a.values, b.values);
      case 238:
        return _mm512_mask_blend_ps(0xFCFC, a.values, b.values);
      case 239:
        return _mm512_mask_blend_ps(0xFCFF, a.values, b.values);
      case 240:
        return _mm512_mask_blend_ps(0xFF00, a.values, b.values);
      case 241:
        return _mm512_mask_blend_ps(0xFF03, a.values, b.values);
      case 242:
        return _mm512_mask_blend_ps(0xFF0C, a.values, b.values);
      case 243:
        return _mm512_mask_blend_ps(0xFF0F, a.values, b.values);
      case 244:
        return _mm512_mask_blend_ps(0xFF30, a.values, b.values);
      case 245:
        return _mm512_mask_blend_ps(0xFF33, a.values, b.values);
      case 246:
        return _mm512_mask_blend_ps(0xFF3C, a.values, b.values);
      case 247:
        return _mm512_mask_blend_ps(0xFF3F, a.values, b.values);
      case 248:
        return _mm512_mask_blend_ps(0xFFC0, a.values, b.values);
      case 249:
        return _mm512_mask_blend_ps(0xFFC3, a.values, b.values);
      case 250:
        return _mm512_mask_blend_ps(0xFFCC, a.values, b.values);
      case 251:
        return _mm512_mask_blend_ps(0xFFCF, a.values, b.values);
      case 252:
        return _mm512_mask_blend_ps(0xFFF0, a.values, b.values);
      case 253:
        return _mm512_mask_blend_ps(0xFFF3, a.values, b.values);
      case 254:
        return _mm512_mask_blend_ps(0xFFFC, a.values, b.values);
    }
    return b;
  }
  static Vectorized<c10::complex<float>> blendv(const Vectorized<c10::complex<float>>& a,
                                               const Vectorized<c10::complex<float>>& b,
                                               const Vectorized<c10::complex<float>>& mask) {
    // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_ = _mm512_unpacklo_ps(mask.values, mask.values);
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask_), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_blend_ps(mmask, a.values, b.values);
  }
  template<typename step_t>
  static Vectorized<c10::complex<float>> arange(c10::complex<float> base = 0.,
                                               step_t step = static_cast<step_t>(1)) {
    return Vectorized<c10::complex<float>>(base,
                                        base + step,
                                        base + c10::complex<float>(2)*step,
                                        base + c10::complex<float>(3)*step,
                                        base + c10::complex<float>(4)*step,
                                        base + c10::complex<float>(5)*step,
                                        base + c10::complex<float>(6)*step,
                                        base + c10::complex<float>(7)*step);
  }
  static Vectorized<c10::complex<float>> set(const Vectorized<c10::complex<float>>& a,
                                            const Vectorized<c10::complex<float>>& b,
                            int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }
    return b;
  }
  static Vectorized<c10::complex<float>> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));

    __at_align__ float tmp_values[2*size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < 2*size(); ++i) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const float*>(ptr),
        count * sizeof(c10::complex<float>));
    return _mm512_load_ps(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[2*size()];
      _mm512_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<float>));
    }
  }
  // AVX512 doesn't have horizontal add & horizontal sub instructions.
  // TODO: hadd_pd() & hsub_pd() may have scope for improvement.
  static inline __m512 hadd_ps(__m512 a, __m512 b) {
  __m512i idx1 = _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0);
  __m512i idx2 = _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1);
  return _mm512_add_ps(_mm512_mask_permutex2var_ps(a, 0xffff, idx1, b),
                       _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b));
  }
  static inline __m512 hsub_ps(__m512 a, __m512 b) {
  __m512i idx1 = _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0);
  __m512i idx2 = _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1);
  return _mm512_sub_ps(_mm512_mask_permutex2var_ps(a, 0xffff, idx1, b),
                       _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b));
  }
  const c10::complex<float>& operator[](int idx) const  = delete;
  c10::complex<float>& operator[](int idx) = delete;
  Vectorized<c10::complex<float>> map(c10::complex<float> (*const f)(const c10::complex<float> &)) const {
    __at_align__ c10::complex<float> tmp[size()];
    store(tmp);
    for (int i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  __m512 abs_2_() const {
    auto val_2 = _mm512_mul_ps(values, values);     // a*a     b*b
    auto ret = hadd_ps(val_2, val_2);        // a*a+b*b a*a+b*b
    return ret;
  }
  __m512 abs_() const {
    return _mm512_sqrt_ps(abs_2_());                // abs     abs
  }
  Vectorized<c10::complex<float>> abs() const {
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    return _mm512_and_ps(abs_(), real_mask);        // abs     0
  }
  __m512 angle_() const {
    //angle = atan2(b/a)
    auto b_a = _mm512_permute_ps(values, 0xB1);     // b        a
    return Sleef_atan2f16_u10(values, b_a);          // 90-angle angle
  }
  Vectorized<c10::complex<float>> angle() const {
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    auto angle = _mm512_permute_ps(angle_(), 0xB1); // angle    90-angle
    return _mm512_and_ps(angle, real_mask);         // angle    0
  }
  Vectorized<c10::complex<float>> sgn() const {
    auto abs = abs_();
    auto zero = _mm512_setzero_ps();
    auto mask = _mm512_cmp_ps_mask(abs, zero, _CMP_EQ_OQ);
    auto abs_val = Vectorized(abs);

    auto div = values / abs_val.values;       // x / abs(x)

    return _mm512_mask_blend_ps(mask, div, zero);
  }
  __m512 real_() const {
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    return _mm512_and_ps(values, real_mask);
  }
  Vectorized<c10::complex<float>> real() const {
    return real_();
  }
  __m512 imag_() const {
    const __m512 imag_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF));
    return _mm512_and_ps(values, imag_mask);
  }
  Vectorized<c10::complex<float>> imag() const {
    return _mm512_permute_ps(imag_(), 0xB1);        //b        a
  }
  __m512 conj_() const {
    const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                            0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    return _mm512_xor_ps(values, sign_mask);        // a       -b
  }
  Vectorized<c10::complex<float>> conj() const {
    return conj_();
  }
  Vectorized<c10::complex<float>> log() const {
    // Most trigonomic ops use the log() op to improve complex number performance.
    return map(std::log);
  }
  Vectorized<c10::complex<float>> log2() const {
    const __m512 log2_ = _mm512_set1_ps(std::log(2));
    return _mm512_div_ps(log(), log2_);
  }
  Vectorized<c10::complex<float>> log10() const {
    const __m512 log10_ = _mm512_set1_ps(std::log(10));
    return _mm512_div_ps(log(), log10_);
  }
  Vectorized<c10::complex<float>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    const __m512 one = _mm512_set1_ps(1);

    auto conj = conj_();
    auto b_a = _mm512_permute_ps(conj, 0xB1);                         //-b        a
    auto ab = _mm512_mul_ps(conj, b_a);                               //-ab       -ab
    auto im = _mm512_add_ps(ab, ab);                                  //-2ab      -2ab

    auto val_2 = _mm512_mul_ps(values, values);                       // a*a      b*b
    auto re = hsub_ps(val_2, _mm512_permute_ps(val_2, 0xB1));  // a*a-b*b  b*b-a*a
    re = _mm512_sub_ps(one, re);

    auto root = Vectorized(_mm512_mask_blend_ps(0xAAAA, re, im)).sqrt();         //sqrt(re + i*im)
    auto ln = Vectorized(_mm512_add_ps(b_a, root)).log();                 //ln(iz + sqrt())
    return Vectorized(_mm512_permute_ps(ln.values, 0xB1)).conj();         //-i*ln()
  }
  Vectorized<c10::complex<float>> acos() const {
    return map(std::acos);
  }
  Vectorized<c10::complex<float>> atan() const;
  Vectorized<c10::complex<float>> atan2(const Vectorized<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> erf() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> erfc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> exp() const {
    //exp(a + bi)
    // = exp(a)*(cos(b) + sin(b)i)
    auto exp = Sleef_expf16_u10(values);                               //exp(a)           exp(b)
    exp = _mm512_mask_blend_ps(0xAAAA, exp, _mm512_permute_ps(exp, 0xB1));   //exp(a)           exp(a)

    auto sin_cos = Sleef_sincosf16_u10(values);                        //[sin(a), cos(a)] [sin(b), cos(b)]
    auto cos_sin = _mm512_mask_blend_ps(0xAAAA, _mm512_permute_ps(sin_cos.y, 0xB1),
                                   sin_cos.x);                  //cos(b)           sin(b)
    return _mm512_mul_ps(exp, cos_sin);
  }
  Vectorized<c10::complex<float>> expm1() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> sin() const {
    return map(std::sin);
  }
  Vectorized<c10::complex<float>> sinh() const {
    return map(std::sinh);
  }
  Vectorized<c10::complex<float>> cos() const {
    return map(std::cos);
  }
  Vectorized<c10::complex<float>> cosh() const {
    return map(std::cosh);
  }
  Vectorized<c10::complex<float>> ceil() const {
    return _mm512_ceil_ps(values);
  }
  Vectorized<c10::complex<float>> floor() const {
    return _mm512_floor_ps(values);
  }
  Vectorized<c10::complex<float>> hypot(const Vectorized<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> igamma(const Vectorized<c10::complex<float>> &x) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> igammac(const Vectorized<c10::complex<float>> &x) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> neg() const {
    auto zero = _mm512_setzero_ps();
    return _mm512_sub_ps(zero, values);
  }
  Vectorized<c10::complex<float>> nextafter(const Vectorized<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> round() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<c10::complex<float>> tan() const {
    return map(std::tan);
  }
  Vectorized<c10::complex<float>> tanh() const {
    return map(std::tanh);
  }
  Vectorized<c10::complex<float>> trunc() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<c10::complex<float>> sqrt() const {
    return map(std::sqrt);
  }
  Vectorized<c10::complex<float>> reciprocal() const;
  Vectorized<c10::complex<float>> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vectorized<c10::complex<float>> pow(const Vectorized<c10::complex<float>> &exp) const {
    __at_align__ c10::complex<float> x_tmp[size()];
    __at_align__ c10::complex<float> y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (int i = 0; i < size(); i++) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<c10::complex<float>> operator==(const Vectorized<c10::complex<float>>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_EQ_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF));
  }
  Vectorized<c10::complex<float>> operator!=(const Vectorized<c10::complex<float>>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_NEQ_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF));
  }
  Vectorized<c10::complex<float>> operator<(const Vectorized<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> operator<=(const Vectorized<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> operator>(const Vectorized<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> operator>=(const Vectorized<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<c10::complex<float>> eq(const Vectorized<c10::complex<float>>& other) const;
  Vectorized<c10::complex<float>> ne(const Vectorized<c10::complex<float>>& other) const;
  Vectorized<c10::complex<float>> lt(const Vectorized<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> le(const Vectorized<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> gt(const Vectorized<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> ge(const Vectorized<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
};

template <> Vectorized<c10::complex<float>> inline operator+(const Vectorized<c10::complex<float>> &a,
                                                            const Vectorized<c10::complex<float>> &b) {
  return _mm512_add_ps(a, b);
}

template <> Vectorized<c10::complex<float>> inline operator-(const Vectorized<c10::complex<float>> &a,
                                                            const Vectorized<c10::complex<float>> &b) {
  return _mm512_sub_ps(a, b);
}

template <> Vectorized<c10::complex<float>> inline operator*(const Vectorized<c10::complex<float>> &a,
                                                            const Vectorized<c10::complex<float>> &b) {
  //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
  const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                          0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm512_mul_ps(a, b);         //ac       bd

  auto d_c = _mm512_permute_ps(b, 0xB1);    //d        c
  d_c = _mm512_xor_ps(sign_mask, d_c);      //d       -c
  auto ad_bc = _mm512_mul_ps(a, d_c);       //ad      -bc

  auto ret = Vectorized<c10::complex<float>>::hsub_ps(ac_bd, ad_bc);  //ac - bd  ad + bc
  return ret;
}

template <> Vectorized<c10::complex<float>> inline operator/(const Vectorized<c10::complex<float>> &a,
                                                            const Vectorized<c10::complex<float>> &b) {
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2()
  //im = (bc - ad)/abs_2()
  const __m512 sign_mask = _mm512_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0,
                                          -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
  auto ac_bd = _mm512_mul_ps(a, b);         //ac       bd

  auto d_c = _mm512_permute_ps(b, 0xB1);    //d        c
  d_c = _mm512_xor_ps(sign_mask, d_c);      //-d       c
  auto ad_bc = _mm512_mul_ps(a, d_c);       //-ad      bc

  auto re_im = Vectorized<c10::complex<float>>::hadd_ps(ac_bd, ad_bc);//ac + bd  bc - ad
  return _mm512_div_ps(re_im, b.abs_2_());
}

// reciprocal. Implement this here so we can use multiplication.
Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::reciprocal() const {
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2() = c/abs_2()
  //im = (bc - ad)/abs_2() = d/abs_2()
  const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                          0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm512_xor_ps(sign_mask, values);    //c       -d
  return _mm512_div_ps(c_d, abs_2_());
}

Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::atan() const {
  // atan(x) = i/2 * ln((i + z)/(i - z))
  const __m512 i = _mm512_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                  0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  const Vectorized i_half = _mm512_setr_ps(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
                                          0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);

  auto sum = Vectorized(_mm512_add_ps(i, values));                      // a        1+b
  auto sub = Vectorized(_mm512_sub_ps(i, values));                      // -a       1-b
  auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
  return i_half*ln;                                                 // i/2*ln()
}

template <>
Vectorized<c10::complex<float>> inline maximum(const Vectorized<c10::complex<float>>& a,
                                              const Vectorized<c10::complex<float>>& b) {
  auto zero_vector = _mm512_set1_epi32(0);
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_LT_OQ);
  auto max = _mm512_mask_blend_ps(mask, a, b);
  // Exploit the fact that all-ones is a NaN.
  auto isnan_mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_UNORD_Q);
  auto isnan = _mm512_mask_set1_epi32(zero_vector, isnan_mask, 0xFFFFFFFF);
  return _mm512_or_ps(max, _mm512_castsi512_ps(isnan));
}

template <>
Vectorized<c10::complex<float>> inline minimum(const Vectorized<c10::complex<float>>& a,
                                              const Vectorized<c10::complex<float>>& b) {
  auto zero_vector = _mm512_set1_epi32(0);
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_GT_OQ);
  auto min = _mm512_mask_blend_ps(mask, a, b);
  // Exploit the fact that all-ones is a NaN.
  auto isnan_mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_UNORD_Q);
  auto isnan = _mm512_mask_set1_epi32(zero_vector, isnan_mask, 0xFFFFFFFF);
  return _mm512_or_ps(min, _mm512_castsi512_ps(isnan));
}

template <>
Vectorized<c10::complex<float>> inline operator&(const Vectorized<c10::complex<float>>& a,
                                                const Vectorized<c10::complex<float>>& b) {
  return _mm512_and_ps(a, b);
}

template <>
Vectorized<c10::complex<float>> inline operator|(const Vectorized<c10::complex<float>>& a,
                                                const Vectorized<c10::complex<float>>& b) {
  return _mm512_or_ps(a, b);
}

template <>
Vectorized<c10::complex<float>> inline operator^(const Vectorized<c10::complex<float>>& a,
                                                const Vectorized<c10::complex<float>>& b) {
  return _mm512_xor_ps(a, b);
}

Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::eq(
    const Vectorized<c10::complex<float>>& other) const {
  return (*this == other) & Vectorized<c10::complex<float>>(_mm512_set1_ps(1.0f));
}

Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::ne(
    const Vectorized<c10::complex<float>>& other) const {
  return (*this != other) & Vectorized<c10::complex<float>>(_mm512_set1_ps(1.0f));
}

#endif

}}}

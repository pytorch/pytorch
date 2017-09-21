from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import sys


def unroll(uf, IndexType, InType, OutType, use_weights, isa):

    def sizeof(InType):
        size = 0
        if InType == "float":
            size = 4
        elif InType == "float16":
            size = 2
        elif InType == "uint8_t":
            size = 1
        else:
            assert False

        return size

    def compute(regid, InType, use_weights, isa, prefetch):
        code = []

        if InType == "float":
            code.append("vop%d = _mm256_fmadd_ps(vwgt,  \
                  _mm256_loadu_ps(ip + (%d)), vop%d);" % (regid, regid, regid))

        elif InType == "float16":
            code.append("vop%d = _mm256_fmadd_ps(vwgt,  \
                   _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ip + (%d)))), \
                   vop%d);"
                        % (regid, regid, regid))
        elif InType == "uint8_t":
            code.append("vop%d = _mm256_fmadd_ps(vwgt,  \
                   _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ip + (%d))))), \
                   _mm256_add_ps(vop%d, vbio));"
                        % (regid, regid, regid))
        else:
            assert False


        if prefetch == True:
            code.append("_mm_prefetch((&ip_next_T0[%d]), _MM_HINT_T0);" % (regid))
        else:
            code.append("// skip unecassery prefetch of (&ip_next_T0[%d])" % (regid))

        return code

    code = []
    code.append("// unrolling " + str(uf) + " times")
    code.append(IndexType + " dataInd = 0;")
    code.append("for (" + IndexType +
                " rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {")
    code.append(OutType + " *op = &out[rangeIndex * block_size];")
    for i in range(0, uf):
        j = 8 * i
        code.append("__m256 vop" + str(j) + " = _mm256_setzero_ps();")

    # inner loop
    code.append("for (" + IndexType +
                " start = dataInd; dataInd < start + lengths[rangeIndex]; ++dataInd) {")
    code.append("const  " + IndexType + " idx = indices[dataInd];")

    if InType == "uint8_t":
        code.append(OutType + " wgt = 1.f;")
        code.append(OutType + " bio;")
        code.append("if (weights) {")
        code.append("wgt = weights[dataInd];")
        code.append("}")
        code.append("bio = wgt * scale_bias[2 * indices[dataInd] + 1];");
        code.append("wgt = wgt * scale_bias[2 * indices[dataInd]];");
        code.append("__m256 vbio = _mm256_set1_ps(bio);")
    else:
        code.append(OutType + " wgt = 1.f;")
        code.append("if (weights) {")
        code.append("wgt = weights[dataInd];")
        code.append("}")
    code.append("__m256 vwgt = _mm256_set1_ps(wgt);")

    code.append("const  " + InType + " *ip = &input[idx * block_size];")
    code.append("const  " + IndexType +
                " next_T0 = (dataInd < index_size - prefdist_T0) ? (dataInd + prefdist_T0) : dataInd;");
    code.append("const  " + IndexType + " idx_pref_T0 = indices[next_T0];")
    code.append(
        "CAFFE_ENFORCE(idx >=0 && idx_pref_T0 >= 0 && idx < data_size && idx_pref_T0 < data_size);")
    code.append("const  " + InType +
                " *ip_next_T0 = &input[idx_pref_T0 * block_size];")

    for i in range(0, uf):
        j = 8 * i
        cachelinesize = 64
        byteoffset = sizeof(InType) * j
        prefetch = ((byteoffset % cachelinesize) == 0)
        code.extend(compute(j, InType, use_weights, isa, prefetch))
    code.append("}")

    code.append("if (normalize_by_lengths == false) {")
    for i in range(0, uf):
        j = 8 * i
        code.append(
            "_mm256_storeu_ps(&op[" + str(j) + "], vop" + str(j) + ");")
    code.append("} else if (lengths[rangeIndex]) {")
    # inv of length
    code.append(
        "__m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);")
    for i in range(0, uf):
        j = 8 * i
        code.append(
            "_mm256_storeu_ps(&op[" + str(j) + "], _mm256_mul_ps(" + "vop" + str(j) + ", vlen_inv));")
    code.append("}")

    code.append("}")
    return code


def generic(IndexType, InType, OutType, use_weights, isa):

    def compute(InType, use_weights, isa):
        code = []
        if InType == "float":
            code.append("_mm256_storeu_ps(&op[j], \
                                 _mm256_fmadd_ps(vwgt,_mm256_loadu_ps(&ip[j]), _mm256_loadu_ps(&op[j])) \
                                   );")
        elif InType == "float16":
            code.append("_mm256_storeu_ps(&op[j], \
                   _mm256_fmadd_ps(vwgt, \
                     _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&ip[j]))), _mm256_loadu_ps(&op[j])) \
                                   );")
        elif InType == "uint8_t":
            code.append("_mm256_storeu_ps(&op[j], \
                   _mm256_fmadd_ps(vwgt, \
                     _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&ip[j])))), \
                     _mm256_add_ps(_mm256_loadu_ps(&op[j]), vbio) ) \
                                   );")
        else:
            assert False


        code.append("_mm_prefetch((&ip_next_T0[j]), _MM_HINT_T0);")

        return code

    code = []
    code.append(IndexType + " dataInd = 0;")
    code.append("for (" + IndexType +
                " rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {")
    code.append(OutType + " *op = &out[rangeIndex * block_size];")

    # initialize to 0
    code.append("TIndex j = 0;")
    code.append("for(; j + 8 <= block_size; j += 8) {")
    code.append("_mm256_storeu_ps(op + j, _mm256_setzero_ps());")
    code.append("}")
    code.append("for(; j < block_size; j++) {")
    code.append("op[j] = 0.0f;")
    code.append("}")

    # inner loop
    code.append("for (" + IndexType +
                " start = dataInd; dataInd < start + lengths[rangeIndex]; ++dataInd) {")
    code.append("const  " + IndexType + " idx = indices[dataInd];")

    if InType == "uint8_t":
        code.append(OutType + " wgt = 1.f;")
        code.append(OutType + " bio;")
        code.append("if (weights) {")
        code.append("wgt = weights[dataInd];")
        code.append("}")
        code.append("assert (scale_bias);")
        code.append("bio = wgt * scale_bias[2 * indices[dataInd] + 1];");
        code.append("wgt = wgt * scale_bias[2 * indices[dataInd]];");
        code.append("__m256 vbio = _mm256_set1_ps(bio);")
    else:
        code.append(OutType + " wgt = 1.f;")
        code.append("if (weights) {")
        code.append("wgt = weights[dataInd];")
        code.append("}")
    code.append("__m256 vwgt = _mm256_set1_ps(wgt);")

    code.append("const  " + InType + " *ip = &input[idx * block_size];")
    code.append("const  " + IndexType +
                " next_T0 = (dataInd < index_size - prefdist_T0) ? (dataInd + prefdist_T0) : dataInd;");
    code.append("const  " + IndexType + " idx_pref_T0 = indices[next_T0];")
    code.append(
        "CAFFE_ENFORCE(idx >=0 && idx_pref_T0 >= 0 && idx < data_size && idx_pref_T0 < data_size);")
    code.append("const  " + InType +
                " *ip_next_T0 = &input[idx_pref_T0 * block_size];")

    # compute and store main loop
    code.append("j = 0;")
    code.append("for(; j + 8 <= block_size; j += 8) {")
    code.extend(compute(InType, use_weights, isa))
    code.append("}")
    # leftover
    if InType == "float16":
        #code.append("float16 vtmp1[8] __attribute__((aligned(64)));")
        code.append("float16 vtmp1[8] CAFFE2_ALIGNED(64);")
    code.append("for(; j < block_size; j++) {")
    if InType == "float":
        code.append("op[j] += wgt * ip[j];")
    elif InType == "float16":
        code.append("vtmp1[0] = ip[j];")
        code.append("__m256 vtmp2 = _mm256_cvtph_ps(*((__m128i*)vtmp1));")
        code.append("op[j] += wgt * ((float*)(&vtmp2))[0];")
    elif InType == "uint8_t":
        code.append("op[j] += wgt * ((float)ip[j]) + bio;")
    else:
        assert False

    code.append("}")

    code.append("}")

    code.append("if (normalize_by_lengths && lengths[rangeIndex]) {")
    code.append("float len_inv = 1.0f / lengths[rangeIndex];")
    code.append("__m256 vlen_inv = _mm256_set1_ps(len_inv);")
    code.append("j = 0;")
    code.append("for(; j + 8 <= block_size; j += 8) {")
    code.append(
        "_mm256_storeu_ps(&op[j], _mm256_mul_ps(_mm256_loadu_ps(&op[j]), vlen_inv));")
    code.append("}")
    code.append("for(; j < block_size; j++) {")
    code.append("op[j] = len_inv * op[j];")
    code.append("}")

    code.append("}")

    code.append("}")
    return code



# start main code
parser = argparse.ArgumentParser()
parser.add_argument('-f', nargs=1, help="file name")
opts = parser.parse_args()
filename = "embedding_lookup_avx2.cc"
if opts.f:
    filename = (opts.f)[0]
fout = open(filename, 'w')

options = [["int32_t", "float",   "float"],
           ["int64_t", "float",   "float"],
           ["int32_t", "float16", "float"],
           ["int64_t", "float16", "float"],
           ["int32_t", "uint8_t",  "float"],
           ["int64_t", "uint8_t",  "float"],
          ]

code = []
# includes
code.append("//// --------------------------")
code.append("//// ATTENTION:                ")
code.append("//// THIS CODE IS AUTOGENERATED")
code.append("//// BY %s                     " % (sys.argv[0]))
code.append("//// DO NOT MODIFY!!!          ")
code.append("//// --------------------------\n\n")

code.append("#include \"caffe2/core/types.h\"")
code.append("#include \"caffe2/core/common.h\"")
code.append("#include <immintrin.h>")
code.append("\n")

code.append("namespace caffe2 {\n")
for o in options:
    [IndexType, InType, OutType] = o

    fn = "void EmbeddingLookup_" + IndexType + \
        "_" + InType + "_" + OutType + "__avx2_fma"
    code.append(fn + "(")
    code.append("const TIndex block_size,")
    code.append("const TIndex output_size,")
    code.append("const TIndex index_size,")
    code.append("const TIndex data_size,")
    code.append("const " + InType + "* input,")
    code.append("const " + IndexType + "* indices,")
    code.append("const int* lengths,")
    code.append("const float* weights,")
    code.append("const float* scale_bias,")
    code.append("bool normalize_by_lengths,")
    code.append(OutType + "* out)")

    code.append("{")
    code.append("const " + IndexType + " prefdist_T0 = 16;")
    #code.append("printf(\"calling " + fn + "\\n\");");
    if InType != "uint8_t":
        code.append("CAFFE_ENFORCE(scale_bias == nullptr, \"scale_bias must be nullptr\");");
    else:
        code.append("CAFFE_ENFORCE(scale_bias != nullptr, \"scale_bias must not be nullptr\");");

    code.append("if (block_size == 128) {")
    code.extend(unroll(16, IndexType, InType, OutType, True, "AVX2"))
    code.append("} else if (block_size == 64) {")
    code.extend(unroll(8, IndexType, InType, OutType, True, "AVX2"))
    code.append("} else if (block_size == 32) {")
    code.extend(unroll(4, IndexType, InType, OutType, True, "AVX2"))
    code.append("} else if (block_size == 16) {")
    code.extend(unroll(2, IndexType, InType, OutType, True, "AVX2"))
    code.append("} else {")
    code.append("// generic code")
    code.extend(generic(IndexType, InType, OutType, True, "AVX2"))
    code.append("}")


    code.append("}")

    code.append("\n")
code.append("} // namespace caffe2")

for c in code:
    #print(c, file = fout)
    fout.write(c + "\n")
fout.close()


print("Created " + filename)

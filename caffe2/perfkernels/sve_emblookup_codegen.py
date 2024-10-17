# mypy: allow-untyped-defs
import argparse
import sys

# Unroll loops when block_size is a multiple of vector length.
def unroll(num_unrolls, IndexType, InType, OutType, use_weights):
    def compute(regid, InType, use_weights):
        code = []

        if InType == "float":
            code.append(
                f"        vsum{regid} =\n"
                "            svmad_f32_x("
                f"svAll, vwgt, svld1_f32(svAll, &ip[{regid} * vLen]),"
                f" vsum{regid});"
            )
        elif InType == "at::Half":
            code.append(
                f"        vsum{regid} = svmad_f32_x(\n"
                "            svAll,\n"
                "            vwgt,\n"
                "            svcvt_f32_f16_x(\n"
                "                svAll,\n"
                "                svreinterpret_f16_u32(svld1uh_u32(\n"
                "                    svAll, reinterpret_cast<const uint16_t*>("
                f"&ip[{regid} * vLen])))),\n"  # noqa
                f"            vsum{regid});"
            )
        elif InType == "at::BFloat16":
            code.append(
                f"        vsum{regid} = svmad_f32_x(\n"
                "            svAll,\n"
                "            vwgt,\n"
                "            svreinterpret_f32_u32(svlsl_n_u32_x(\n"
                "                svAll,\n"
                "                svld1uh_u32(\n"
                "                    svAll, reinterpret_cast<const uint16_t*>("
                f"&ip[{regid} * vLen])),\n"
                "                16)),\n"  # noqa
                f"            vsum{regid});"
            )
        elif InType == "uint8_t":
            code.append(
                f"        vsum{regid} = svmad_f32_x(\n"
                "            svAll,\n"
                "            vwgt,\n"
                "            svcvt_f32_u32_x(svAll,"
                f" svld1ub_u32(svAll, &ip[{regid} * vLen])),\n"  # noqa
                f"            svadd_f32_x(svAll, vsum{regid}, vbio));"
            )
        else:
            raise ValueError(f"Unknown datatype \"{InType}\"")

        return code

    code = []
    code.append(f"    // unrolling {num_unrolls} times")

    code.append("    for (int64_t i = 0; i < output_size; ++i) {")

    code.append("      " + OutType + "* const op = &out[i * block_size];")
    code.append(
        "      if (pos != offsets[i] - offsets[0]) {\n"
        + "        return false;\n"
        + "      }"
    )

    # Initialise vector sum registers
    for i in range(num_unrolls):
        code.append(f"      svfloat32_t vsum{i} = svdup_n_f32(0);")

    # inner loop
    code.append("""\
      int64_t start_offset = offsets[i];
      int64_t end_offset = offsets[i + 1];""")
    code.append(
        "      for ("
        + "int64_t"
        + " j = start_offset; j < end_offset; ++j) {"  # noqa
    )

    code.append("        const auto idx = indices[pos];")
    code.append(
        "        if (idx < 0 || idx >= data_size) {\n"
        + "          return false;\n"
        + "        }"
    )

    if InType == "uint8_t":
        code.append("        " + OutType + " wgt = 1.f;")
        code.append("        " + OutType + " bio{};")
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (j - start_offset) : pos];"  # noqa
        )
        code.append("        }")
        code.append("        if (scale_bias) {")
        code.append("          bio = wgt * scale_bias[2 * idx + 1];")
        code.append("          wgt = wgt * scale_bias[2 * idx];")
        code.append("        }")
        code.append("        svfloat32_t vbio = svdup_n_f32(bio);")
    else:
        code.append("        " + OutType + " wgt = 1.f;")
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (j - start_offset) : pos];"  # noqa
        )
        code.append("        }")

    code.append("        const svfloat32_t vwgt = svdup_n_f32(wgt);")
    code.append(f"        const {InType}* const ip = &input[idx * block_size];")
    code.append("        // weight * input + out")

    for i in range(num_unrolls):
        code.extend(compute(i, InType, use_weights))

    code.append("        ++pos;")
    code.append("      }")

    code.append("      // Normalisation")
    code.append("      const int64_t length = end_offset - start_offset;")
    code.append("      if (normalize_by_lengths && length != 0) {")
    code.append("        const float len_inv = 1.0f / length;")
    code.append("        const svfloat32_t vlen_inv = svdup_n_f32(len_inv);")

    for i in range(num_unrolls):
        code.append(f"        svst1_f32(svAll, &op[{i} * vLen],"
                    + f" svmul_f32_x(svAll, vsum{i}, vlen_inv));")

    code.append("      } else {")
    # inv of length
    for i in range(num_unrolls):
        code.append(f"        svst1_f32(svAll, &op[{i} * vLen], vsum{i});")

    code.append("      }")
    code.append("    }")
    return code


# Handle the case where block_size is not a multiple of vector length.
def generic(IndexType, InType, OutType, use_weights):
    def compute(InType, use_weights):
        code = []
        if InType == "float":
            code.append(
                "          svst1_f32(\n"
                "              pg,\n"
                "              &op[k],\n"
                "              svmad_f32_x(\n"
                "                  pg, vwgt, svld1_f32(pg, &ip[k]),"
                " svld1_f32(pg, &op[k])));"
            )
        elif InType == "at::Half":
            code.append(
                "          svst1_f32(\n"
                "              pg,\n"
                "              &op[k],\n"
                "              svmad_f32_x(\n"
                "                  pg,\n"
                "                  vwgt,\n"
                "                  svcvt_f32_f16_x(\n"
                "                      pg,\n"
                "                      svreinterpret_f16_u32(svld1uh_u32(\n"
                "                          pg,"
                " reinterpret_cast<const uint16_t*>(&ip[k])))),\n"
                "                  svld1_f32(pg, &op[k])));"
            )
        elif InType == "at::BFloat16":
            code.append(
                "          svst1_f32(\n"
                "              pg,\n"
                "              &op[k],\n"
                "              svmad_f32_x(\n"
                "                  pg,\n"
                "                  vwgt,\n"
                "                  svreinterpret_f32_u32(svlsl_n_u32_x(\n"
                "                      pg,\n"
                "                      svld1uh_u32(\n"
                "                          pg,"
                " reinterpret_cast<const uint16_t*>(&ip[k])),\n"
                "                      16)),\n"
                "                  svld1_f32(pg, &op[k])));"
            )
        elif InType == "uint8_t":
            code.append(
                "          svst1_f32(\n"
                "              pg,\n"
                "              &op[k],\n"
                "              svmad_f32_x(\n"
                "                  pg,\n"
                "                  vwgt,\n"
                "                  svcvt_f32_u32_x(pg,"
                " svld1ub_u32(pg, &ip[k])),\n"  # noqa
                "                  svadd_f32_x(pg,"
                " svld1_f32(pg, &op[k]), vbio)));"
            )
        else:
            raise ValueError(f"Unknown datatype \"{InType}\"")

        return code

    code = []

    code.append(
        "    for (int64_t i = 0; i < output_size; ++i) {"
    )

    code.append("      " + OutType + "* const op = &out[i * block_size];")

    # initialize to 0
    code.append("      memset(op, 0, sizeof(float) * block_size);")

    # inner loop
    code.append(
        "      if (pos != offsets[i] - offsets[0]) {\n"
        + "        return false;\n"
        + "      }"
    )
    code.append(
        "      int64_t start_offset = offsets[i];\n"
        + "      int64_t end_offset = offsets[i + 1];"
    )
    code.append(
        "      for ("
        + "int64_t"
        + " j = start_offset; j < end_offset; ++j) {"  # noqa
    )

    code.append("        const auto idx = indices[pos];")
    code.append(
        "        if (idx < 0 || idx >= data_size) {\n"
        + "          return false;\n"
        + "        }"
    )

    if InType == "uint8_t":
        code.append("        // unimplemented")
        code.append("        " + OutType + " wgt = 1.f;")
        code.append("        " + OutType + " bio{};")
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (j - start_offset) : pos];"  # noqa
        )
        code.append("        }")
        code.append("        if (scale_bias) {")
        code.append("          bio = wgt * scale_bias[2 * idx + 1];")
        code.append("          wgt = wgt * scale_bias[2 * idx];")
        code.append("        }")
        code.append("        svfloat32_t vbio = svdup_n_f32(bio);")
    else:
        code.append("        " + OutType + " wgt = 1.f;")
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (j - start_offset) : pos];"  # noqa
        )
        code.append("        }")

    code.append("        const svfloat32_t vwgt = svdup_n_f32(wgt);")
    code.append(f"        const {InType}* ip = &input[idx * block_size];")

    # compute and store main loop
    code.append("        svbool_t pg;")
    code.append("        for (int64_t k = 0;")
    code.append("             svptest_first(svAll, pg = svwhilelt_b32_s64("
                + "k, block_size));")
    code.append("             k += vLen) {")
    code.extend(compute(InType, use_weights))
    code.append("        }\n")
    code.append("        ++pos;")
    code.append("      }")

    code.append("      const int64_t length = end_offset - start_offset;\n")
    code.append("      if (normalize_by_lengths && length != 0) {")
    code.append("        const float len_inv = 1.0f / length;")
    code.append("        svfloat32_t vlen_inv = svdup_n_f32(len_inv);")
    code.append("        svbool_t pg;")
    code.append("        for (int64_t j = 0;\n"
                "             svptest_first(svAll, pg = svwhilelt_b32_s64("
                "j, block_size));")
    code.append("             j += vLen) {")
    code.append(
        "          svst1_f32(\n"
        "              pg, &op[j], svmul_f32_x(pg, svld1_f32(pg, &op[j]), vlen_inv));"
    )
    code.append("        }")
    code.append("      }")
    code.append("    }")
    return code

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="file name")
    opts = parser.parse_args()
    if opts.filename:
        filename = opts.filename
    else:
        filename = "embedding_lookup_idx_sve.cc"

    options = [
        ["int32_t", "int32_t", "float", "float", "float", "float"],
        ["int64_t", "int64_t", "float", "float", "float", "float"],
        ["int32_t", "int32_t", "half", "at::Half", "float", "float"],
        ["int64_t", "int64_t", "half", "at::Half", "float", "float"],
        ["int32_t", "int32_t", "bfloat16", "at::BFloat16", "float", "float"],
        ["int64_t", "int64_t", "bfloat16", "at::BFloat16", "float", "float"],
        ["int32_t", "int32_t", "uint8_t", "uint8_t", "float", "float"],
        ["int64_t", "int64_t", "uint8_t", "uint8_t", "float", "float"],
    ]

    code = []
    # includes
    code.append("//// --------------------------")
    code.append("//// ATTENTION:")
    code.append("//// THIS CODE IS AUTOGENERATED")
    code.append(f"//// BY {' '.join(sys.argv)}")
    code.append("//// DO NOT MODIFY!!!")
    code.append("//// --------------------------\n")

    code.append("#include <arm_sve.h>")
    code.append("#include <c10/util/BFloat16.h>")
    code.append("#include <c10/util/Half.h>")
    code.append("#include <cstdint>")
    code.append("#include <cstring>")

    code.append("namespace caffe2 {\n")
    for o in options:
        [IndexTypeName, IndexType, InTypeName, InType, OutTypeName, OutType] = o

        code.append("template <bool IS_WEIGHT_POSITIONAL>")
        fn_base = f"EmbeddingLookupIdx_{IndexTypeName}_{InTypeName}_{OutTypeName}"

        suffix = "__sve"
        fn = "static bool " + fn_base + suffix
        code.append(fn + "(")

        args = []
        args.append("    const int64_t block_size,")
        args.append("    const int64_t output_size,")
        args.append("    const int64_t index_size,")
        args.append("    const int64_t data_size,")
        args.append("    const " + InType + "* input,")
        args.append("    const " + IndexType + "* indices,")
        args.append("    const " + IndexType + "* offsets,")
        args.append("    const float* weights,")
        args.append("    const float* scale_bias,")
        args.append("    bool normalize_by_lengths,")
        args.append("    " + OutType + "* out) {")
        code += args

        code.append("  const svbool_t svAll = svptrue_b32();")
        code.append("  const auto vLen = static_cast<int64_t>(svcntw());")
        code.append("  int64_t pos = 0;")

        code.append("  if (block_size == 32 * vLen) {")
        code += unroll(32, IndexType, InType, OutType, True)
        code.append("  } else if (block_size == 16 * vLen) {")
        code += unroll(16, IndexType, InType, OutType, True)
        code.append("  } else if (block_size == 8 * vLen) {")
        code += unroll(8, IndexType, InType, OutType, True)
        code.append("  } else if (block_size == 4 * vLen) {")
        code += unroll(4, IndexType, InType, OutType, True)
        code.append("  } else if (block_size == 2 * vLen) {")
        code += unroll(2, IndexType, InType, OutType, True)
        code.append("  } else {")
        code.append("    // generic code:")
        code += generic(IndexType, InType, OutType, True)
        code.append("  }")
        code.append("  return pos == index_size;")

        code.append("}")

        for is_weight_positional in ["false", "true"]:
            code.append("bool " + fn_base + "_" + is_weight_positional + suffix + "(")
            code += args

            # Resolve the Lint warnings: Limit of 80 characters in one line.
            extra_space = "\n      "
            ret_string = "  return " + fn_base + suffix \
                    + "<" + is_weight_positional + ">("
            if len(ret_string) <= 80:
                code.append(ret_string)
            else:
                code.append("  return " + fn_base + suffix + "<" + extra_space + is_weight_positional + ">(")

            code.append("      block_size,")
            code.append("      output_size,")
            code.append("      index_size,")
            code.append("      data_size,")
            code.append("      input,")
            code.append("      indices,")
            code.append("      offsets,")
            code.append("      weights,")
            code.append("      scale_bias,")
            code.append("      normalize_by_lengths,")
            code.append("      out);")
            code.append("}")

        code.append("")

    code.append("} // namespace caffe2")

    with open(filename, "w") as fout:
        fout.write("\n".join(code) + "\n")

    print("Created " + filename)

if __name__ == "__main__":
    main()

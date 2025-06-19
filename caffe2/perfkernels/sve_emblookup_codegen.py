# mypy: allow-untyped-defs
import argparse
import sys


# Unroll loops when block_size is a multiple of vector length.
def unroll(num_unrolls, IndexType, InType, OutType):
    def compute_output(num_unrolls, InType, is_main):
        code = []

        pred = "svAll" if is_main else "pg"
        if InType == "float":
            for i in range(num_unrolls):
                code.append(f"        output = svmla_x({pred}, output, svld1(svAll, &ip{i}[k]), wgt{i});")
        elif InType == "at::Half":
            for i in range(num_unrolls):
                code.append(f"        auto input{i} = svcvt_f32_x({pred}, svreinterpret_f16(\n"
                f"          svld1uh_u32({pred}, reinterpret_cast<const uint16_t*>(&ip{i}[k]))));")
            for i in range(num_unrolls):
                code.append(f"        output = svmla_x({pred}, output, input{i}, wgt{i});")
        elif InType == "at::BFloat16":
            for i in range(num_unrolls):
                code.append(f"        auto input{i} = svreinterpret_f32(svlsl_x({pred},\n"
                f"          svld1uh_u32({pred}, reinterpret_cast<const uint16_t*>(&ip{i}[k])), 16));")
            for i in range(num_unrolls):
                code.append(f"        output = svmla_x({pred}, output, input{i}, wgt{i});")
        elif InType == "uint8_t":
            code.append(f"        output = svadd_x({pred}, output, bio);")
            for i in range(num_unrolls):
                code.append(f"        auto input{i} = svcvt_f32_x({pred}, svld1ub_u32({pred}, &ip{i}[k]));")
            for i in range(num_unrolls):
                code.append(f"        output = svmla_x({pred}, output, input{i}, wgt{i});")
        else:
            raise ValueError(f'Unknown datatype "{InType}"')

        return code

    code = []

    if num_unrolls == 1:
        code.append(f"    // tail loop")
        code.append("    if (j < end_offset) {")
    else:
        code.append(f"    // unrolling {num_unrolls} times")
        code.append(f"    while (j + {num_unrolls - 1} < end_offset) {{")
    for i in range(num_unrolls):
        code.append(f"      const auto idx{i} = indices[pos + {i}];")

    # check indices
    for i in range(num_unrolls):
        code.append(
            f"      if (idx{i} < 0 || idx{i} >= data_size) {{\n"
            + "        return false;\n"
            + "      }"
        )

    if InType == "uint8_t":
        for i in range(num_unrolls):
            code.append(f"      {OutType} wgt{i} = 1.f;")
        code.append(f"      {OutType} bio = 0.f;")
    else:
        for i in range(num_unrolls):
            code.append(f"      {OutType} wgt{i} = 1.f;")

    code.append("      if (weights) {")
    for i in range(num_unrolls):
        code.append(f"        wgt{i} = weights[IS_WEIGHT_POSITIONAL ? (j + {i} - start_offset) : pos + {i}];")
    code.append("      }")
    if InType == "uint8_t":
        code.append("      if (scale_bias) {")
        for i in range(num_unrolls):
            code.append(f"        bio += wgt{i} * scale_bias[2 * idx{i} + 1];")
            code.append(f"        wgt{i} = wgt{i} * scale_bias[2 * idx{i}];")
        code.append("      }")

    for i in range(num_unrolls):
        code.append(f"      const {InType}* const ip{i} = &input[idx{i} * block_size];")

    # compute and store
    code.append("      svbool_t pg;")
    code.append("      int64_t k = 0;")
    # main loop
    code.append("      while (k + vLen - 1 < block_size) {")
    code.append("        auto output = svld1(svAll, &op[k]);")
    code.extend(compute_output(num_unrolls, InType, True))
    code.append("        svst1(svAll, &op[k], output);")
    code.append("        k += vLen;")
    code.append("      }")
    # tail loop
    code.append("      if (k < block_size) {")
    code.append("        pg = svwhilelt_b32_s64(k, block_size);")
    code.append("        auto output = svld1(pg, &op[k]);")
    code.extend(compute_output(num_unrolls, InType, False))
    code.append("        svst1(pg, &op[k], output);")
    code.append("        k += vLen;")
    code.append("      }")
    if num_unrolls == 1:
        code.append("      pos ++;")
    else:
        code.append(f"      j += {num_unrolls};")
        code.append(f"      pos += {num_unrolls};")

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

        code.append("  for (int64_t i = 0; i < output_size; ++i) {")
        code.append("    " + OutType + "* const op = &out[i * block_size];")

        # initialize to 0
        code.append("    memset(op, 0, sizeof(float) * block_size);")

        # inner loop
        code.append(
            "    if (pos != offsets[i] - offsets[0]) {\n"
            + "      return false;\n"
            + "    }"
        )
        code.append(
            "    int64_t start_offset = offsets[i];\n"
            + "    int64_t end_offset = offsets[i + 1];"
        )
        code.append("    int64_t j = start_offset;")

        code += unroll(16, IndexType, InType, OutType)
        code += unroll(8, IndexType, InType, OutType)
        code += unroll(4, IndexType, InType, OutType)
        code += unroll(2, IndexType, InType, OutType)
        code += unroll(1, IndexType, InType, OutType)

        code.append("    const int64_t length = end_offset - start_offset;\n")
        code.append("    if (normalize_by_lengths && length != 0) {")
        code.append("      const float len_inv = 1.0f / length;")
        code.append("      svbool_t pg;")
        code.append("      int64_t j = 0;")
        code.append("      while (j + vLen - 1 < block_size) {")
        code.append("        svst1(svAll, &op[j], svmul_x(svAll, svld1(svAll, &op[j]), len_inv));")
        code.append("        j += vLen;")
        code.append("      }")
        code.append("      if (j < block_size) {")
        code.append("        pg = svwhilelt_b32_s64(j, block_size);")
        code.append("        svst1(pg, &op[j], svmul_x(pg, svld1(pg, &op[j]), len_inv));")
        code.append("      }")
        code.append("    }")

        code.append("  }")
        code.append("  return pos == index_size;")
        code.append("}")

        for is_weight_positional in ["false", "true"]:
            code.append("bool " + fn_base + "_" + is_weight_positional + suffix + "(")
            code += args

            # Resolve the Lint warnings: Limit of 80 characters in one line.
            extra_space = "\n      "
            ret_string = (
                "  return " + fn_base + suffix + "<" + is_weight_positional + ">("
            )
            if len(ret_string) <= 80:
                code.append(ret_string)
            else:
                code.append(
                    "  return "
                    + fn_base
                    + suffix
                    + "<"
                    + extra_space
                    + is_weight_positional
                    + ">("
                )

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

#!/bin/bash
set -ex
rm -f *.cu
IFS=","

# BACKWARD
kernel="BACKWARD"
kernel_lower=`echo "\$kernel" | awk '{print tolower($0)}'`
for aligned in "false" "true"; do
    for maxk in 64 128 ""; do
        for dtype_name in "f32" "f16" "bf16"; do
            case "$dtype_name" in
                "f32") dtype="float" ;;
                "f16") dtype="cutlass::half_t" ;;
                "bf16") dtype="cutlass::bfloat16_t" ;;
            esac
            [[ $aligned = "true" ]] && s="_aligned" || s=""
            [[ $maxk = "" ]] && s="${s}" || s="${s}_k$maxk"
            [[ $maxk = "" ]] && maxk_code="" || maxk_code=", $maxk"
            FNAME="${kernel_lower}_${dtype_name}${s}.cu"
            echo $FNAME
            cat <<EOF > $FNAME
// This file is auto-generated. See "generate_kernels.sh"
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
EOF
            for sm in 50 70 75 80; do
                echo "INSTANTIATE_ATTENTION_KERNEL_${kernel}_SM${sm}($dtype, $aligned$maxk_code);" >> $FNAME
            done;
        done;
    done;
done

# FORWARD
kernel="FORWARD"
kernel_lower=`echo "\$kernel" | awk '{print tolower($0)}'`
for aligned in "false" "true"; do
    [[ $aligned = "true" ]] && aligned_suffix="_aligned" || aligned_suffix=""
    for dtype_name in "f32" "f16" "bf16"; do
        case "$dtype_name" in
            "f32") dtype="float" ;;
            "f16") dtype="cutlass::half_t" ;;
            "bf16") dtype="cutlass::bfloat16_t" ;;
        esac
        FNAME="${kernel_lower}_${dtype_name}${aligned_suffix}.cu"
        echo $FNAME
        cat <<EOF > $FNAME
// This file is auto-generated. See "generate_kernels.sh"
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
EOF
        for sm in 50 70 75 80; do
            echo "INSTANTIATE_ATTENTION_KERNEL_${kernel}_SM${sm}($dtype, $aligned, 32, 128, true);" >> $FNAME
            echo "INSTANTIATE_ATTENTION_KERNEL_${kernel}_SM${sm}($dtype, $aligned, 32, 128, false);" >> $FNAME
            echo "INSTANTIATE_ATTENTION_KERNEL_${kernel}_SM${sm}($dtype, $aligned, 64, 64, true);" >> $FNAME
        done;
    done;
done

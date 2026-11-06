def select_mkldnn(enabled, disabled):
    return select({
        "DEFAULT": select({
            "DEFAULT": enabled,
            "//caffe2/constraints:caffe2_use_mkldnn_disabled": disabled,
        }),
        "//caffe2/constraints:mkldnn[disabled]": disabled,
    })

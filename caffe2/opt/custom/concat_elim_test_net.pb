name: "train_net"
op {
  input: "float_features:values:keys"
  input: "float_features:values:values"
  input: "ZERO"
  input: "float_features:lengths"
  output: "feature_preproc/feature_sparse_to_dense/float_features_output"
  name: ""
  type: "SparseToDenseMask"
  arg {
    name: "mask"
    ints: 1269
    ints: 51
    ints: 164
    ints: 915
  }
}
op {
  input: "id_list_features:values:values:lengths"
  output: "feature_preproc/feature_sparse_to_dense_auto_0/id_list_ranges"
  name: ""
  type: "LengthsToRanges"
}
op {
  input: "id_list_features:values:keys"
  input: "feature_preproc/feature_sparse_to_dense_auto_0/id_list_ranges"
  input: "ZERO_RANGE"
  input: "id_list_features:lengths"
  output: "feature_preproc/feature_sparse_to_dense_auto_0/id_list_features_ranges"
  name: ""
  type: "SparseToDenseMask"
  arg {
    name: "mask"
    ints: 98
    ints: 38
    ints: 100
    ints: 19
    ints: 9
    ints: 30
  }
}
op {
  input: "id_list_features:values:values:values"
  output: "feature_preproc/feature_sparse_to_dense_auto_0/id_list_features_values"
  name: ""
  type: "Alias"
}
op {
  input: "feature_preproc/feature_sparse_to_dense/float_features_output"
  input: "feature_preproc/feature_proc_group_0_starts"
  input: "feature_preproc/feature_proc_group_0_ends"
  output: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_0"
  name: ""
  type: "Slice"
}
op {
  input: "feature_preproc/feature_sparse_to_dense/float_features_output"
  input: "feature_preproc/feature_proc_group_1_starts"
  input: "feature_preproc/feature_proc_group_1_ends"
  output: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_1"
  name: ""
  type: "Slice"
}
op {
  input: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_1"
  output: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_1_logit"
  name: ""
  type: "Logit"
}
op {
  input: "feature_preproc/feature_sparse_to_dense/float_features_output"
  input: "feature_preproc/feature_proc_group_2_starts"
  input: "feature_preproc/feature_proc_group_2_ends"
  output: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_2"
  name: ""
  type: "Slice"
}
op {
  input: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_2"
  output: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_2_int64"
  name: ""
  type: "Cast"
  arg {
    name: "to"
    i: 10
  }
}
op {
  input: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_2_int64"
  input: "feature_preproc/feature_proc_group_2_one_hot_lens"
  input: "feature_preproc/feature_proc_group_2_one_hot_vals"
  output: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_2_one_hot"
  name: ""
  type: "BatchOneHot"
}
op {
  input: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_2_one_hot"
  output: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_2_one_hot_float"
  name: ""
  type: "Cast"
  arg {
    name: "to"
    i: 1
  }
}
op {
  input: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_0"
  input: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_1_logit"
  input: "feature_preproc/feature_sparse_to_dense/float_features_output_slice_2_one_hot_float"
  output: "feature_preproc/feature_proc/output"
  output: "feature_preproc/feature_proc/output_split_info"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "feature_preproc/feature_proc/output"
  input: "feature_preproc/feature_proc_shift_blob"
  output: "feature_preproc/feature_proc/output"
  name: ""
  type: "Add"
  arg {
    name: "broadcast"
    i: 1
  }
}
op {
  input: "feature_preproc/feature_proc/output"
  input: "feature_preproc/feature_proc_scale_blob"
  output: "feature_preproc/feature_proc/output"
  name: ""
  type: "Mul"
  arg {
    name: "broadcast"
    i: 1
  }
}
op {
  input: "feature_preproc/feature_proc/output"
  output: "feature_preproc/feature_proc/output"
  name: ""
  type: "ReplaceNaN"
  arg {
    name: "value"
    f: 0.0
  }
}
op {
  input: "feature_preproc/feature_proc/output"
  output: "feature_preproc/feature_proc/output"
  name: ""
  type: "Clip"
  arg {
    name: "max"
    f: 10.0
  }
  arg {
    name: "min"
    f: -10.0
  }
}
op {
  input: "feature_preproc/feature_proc/output"
  output: "feature_preproc/feature_proc/output"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/sigrid_transform_transform_instance"
  input: "feature_preproc/feature_sparse_to_dense_auto_0/id_list_features_ranges"
  input: "feature_preproc/feature_sparse_to_dense_auto_0/id_list_features_values"
  input: "feature_preproc/feature_sparse_to_dense/float_features_output"
  output: "feature_preproc/sigrid_transform/range_SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM"
  output: "feature_preproc/sigrid_transform/values_SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM"
  output: "feature_preproc/sigrid_transform/range_SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  output: "feature_preproc/sigrid_transform/values_SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  output: "feature_preproc/sigrid_transform/range_SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  output: "feature_preproc/sigrid_transform/values_SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  name: ""
  type: "SigridTransforms"
}
op {
  input: "feature_preproc/sigrid_transform/values_SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM"
  input: "feature_preproc/sigrid_transform/range_SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM"
  output: "feature_preproc/output_features:SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  output: "feature_preproc/output_features:SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  name: ""
  type: "GatherRanges"
}
op {
  input: "feature_preproc/sigrid_transform/values_SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  input: "feature_preproc/sigrid_transform/range_SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  output: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  name: ""
  type: "GatherRanges"
}
op {
  input: "feature_preproc/sigrid_transform/values_SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  input: "feature_preproc/sigrid_transform/range_SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  output: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  name: ""
  type: "GatherRanges"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  output: "feature_preproc/output_features:SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  output: "feature_preproc/output_features:SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/sigrid_transform_auto_0_transform_instance"
  input: "feature_preproc/feature_sparse_to_dense_auto_0/id_list_features_ranges"
  input: "feature_preproc/feature_sparse_to_dense_auto_0/id_list_features_values"
  input: "feature_preproc/feature_sparse_to_dense/float_features_output"
  output: "feature_preproc/sigrid_transform_auto_0/range_SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM"
  output: "feature_preproc/sigrid_transform_auto_0/values_SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM"
  output: "feature_preproc/sigrid_transform_auto_0/range_SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  output: "feature_preproc/sigrid_transform_auto_0/values_SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  output: "feature_preproc/sigrid_transform_auto_0/range_SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM"
  output: "feature_preproc/sigrid_transform_auto_0/values_SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM"
  name: ""
  type: "SigridTransforms"
}
op {
  input: "feature_preproc/sigrid_transform_auto_0/values_SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM"
  input: "feature_preproc/sigrid_transform_auto_0/range_SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM"
  output: "feature_preproc/output_features:SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  output: "feature_preproc/output_features:SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  name: ""
  type: "GatherRanges"
}
op {
  input: "feature_preproc/sigrid_transform_auto_0/values_SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  input: "feature_preproc/sigrid_transform_auto_0/range_SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS"
  output: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  name: ""
  type: "GatherRanges"
}
op {
  input: "feature_preproc/sigrid_transform_auto_0/values_SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM"
  input: "feature_preproc/sigrid_transform_auto_0/range_SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM"
  output: "feature_preproc/output_features:SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  output: "feature_preproc/output_features:SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  name: ""
  type: "GatherRanges"
}
op {
  input: "feature_preproc/output_features:SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  output: "feature_preproc/output_features:SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  output: "feature_preproc/output_features:SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  output: "feature_preproc/output_features:SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  name: ""
  type: "StopGradient"
}
op {
  input: "feature_preproc/output_features:SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  output: "feature_preproc/output_features:SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  name: ""
  type: "StopGradient"
}
op {
  input: "label"
  output: "supervision_preproc/Cast/label_float"
  name: ""
  type: "Cast"
  arg {
    name: "to"
    i: 1
  }
}
op {
  input: "feature_preproc/feature_proc/output"
  input: "nested/dense/fc/w"
  input: "nested/dense/fc/b"
  output: "nested/dense/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "nested/dense/fc/output"
  output: "nested/dense/Relu/relu"
  name: ""
  type: "Relu"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/LogMeanExp_reducer"
  name: ""
  type: "SortedSegmentRangeLogMeanExp"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Sum_reducer"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Max_reducer"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/LogMeanExp_reducer"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/w"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/b"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/LogMeanExp_reducer"
  name: ""
  type: "SortedSegmentRangeLogMeanExp"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Sum_reducer"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Max_reducer"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/LogMeanExp_reducer"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/w"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/b"
  output: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Mean_reducer"
  name: ""
  type: "LengthsMean"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Mean_reducer"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Max_reducer"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/w"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/b"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Mean_reducer"
  name: ""
  type: "LengthsMean"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Mean_reducer"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Max_reducer"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/w"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/b"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/LogMeanExp_reducer"
  name: ""
  type: "SortedSegmentRangeLogMeanExp"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/Sum_reducer"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/Max_reducer"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer/LogMeanExp_reducer"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/fc/w"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/fc/b"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/LogMeanExp_reducer"
  name: ""
  type: "SortedSegmentRangeLogMeanExp"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/Sum_reducer"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/Max_reducer"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer/LogMeanExp_reducer"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/fc/w"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/fc/b"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/LogMeanExp_reducer"
  name: ""
  type: "SortedSegmentRangeLogMeanExp"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/Sum_reducer"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/Max_reducer"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer/LogMeanExp_reducer"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/fc/w"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/fc/b"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/sparse_lookup/output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/pack_segments/batch_embedding"
  name: ""
  type: "PackSegments"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/pack_segments/batch_embedding"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/expand_dims/batch_embedding"
  name: ""
  type: "ExpandDims"
  arg {
    name: "dims"
    ints: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/expand_dims/batch_embedding"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/conv_kernel"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/conv_bias"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/output"
  name: ""
  type: "Conv"
  arg {
    name: "kernel_w"
    i: 3
  }
  arg {
    name: "pad_l"
    i: 0
  }
  arg {
    name: "pad_b"
    i: 0
  }
  arg {
    name: "stride_h"
    i: 1
  }
  arg {
    name: "stride_w"
    i: 1
  }
  arg {
    name: "pad_r"
    i: 2
  }
  arg {
    name: "order"
    s: "NHWC"
  }
  arg {
    name: "kernel_h"
    i: 1
  }
  arg {
    name: "pad_t"
    i: 0
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/squeeze/conv_output_squeeze"
  name: ""
  type: "Squeeze"
  arg {
    name: "dims"
    ints: 1
  }
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/squeeze/conv_output_squeeze"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/unpack_segments/conv_output_unpack"
  name: ""
  type: "UnpackSegments"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/unpack_segments/conv_output_unpack"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer/Sum_reducer"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/expand_dims/batch_embedding"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/conv_kernel"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/conv_bias"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/output"
  name: ""
  type: "Conv"
  arg {
    name: "kernel_w"
    i: 5
  }
  arg {
    name: "pad_l"
    i: 0
  }
  arg {
    name: "pad_b"
    i: 0
  }
  arg {
    name: "stride_h"
    i: 1
  }
  arg {
    name: "stride_w"
    i: 1
  }
  arg {
    name: "pad_r"
    i: 4
  }
  arg {
    name: "order"
    s: "NHWC"
  }
  arg {
    name: "kernel_h"
    i: 1
  }
  arg {
    name: "pad_t"
    i: 0
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/squeeze_auto_0/conv_output_squeeze"
  name: ""
  type: "Squeeze"
  arg {
    name: "dims"
    ints: 1
  }
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/squeeze_auto_0/conv_output_squeeze"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/unpack_segments_auto_0/conv_output_unpack"
  name: ""
  type: "UnpackSegments"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/unpack_segments_auto_0/conv_output_unpack"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_auto_0/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_auto_0/Sum_reducer"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output_auto_0"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output_auto_0_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output_auto_0"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/concat/output"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/concat/output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "add_axis"
    i: 0
  }
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/concat/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/w"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/b"
  output: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Mean_reducer"
  name: ""
  type: "LengthsMean"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/LogSumExp_reducer"
  name: ""
  type: "SortedSegmentRangeLogSumExp"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Mean_reducer"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/Max_reducer"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer/LogSumExp_reducer"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/w"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/b"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  name: ""
  type: "LengthsToSegmentIds"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Mean_reducer"
  name: ""
  type: "LengthsMean"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Max_reducer"
  name: ""
  type: "SortedSegmentRangeMax"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths_seg_ids"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/LogSumExp_reducer"
  name: ""
  type: "SortedSegmentRangeLogSumExp"
  engine: "fp16"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Mean_reducer"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/Max_reducer"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer/LogSumExp_reducer"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/w"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/b"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/PackRNNSequence/lstm_input"
  name: ""
  type: "PackRNNSequence"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/PackRNNSequence/lstm_input"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/AddPadding/rnn_input_pad"
  name: ""
  type: "AddPadding"
  arg {
    name: "padding_width"
    i: 0
  }
  arg {
    name: "end_padding_width"
    i: 1
  }
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Copy/seq_lengths_copy"
  name: ""
  type: "Copy"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Copy/seq_lengths_copy"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/ExpandDims/expanded_seq_lengths"
  name: ""
  type: "ExpandDims"
  arg {
    name: "dims"
    ints: 0
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/AddPadding/rnn_input_pad"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/i2h_w"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/i2h_b"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/i2h"
  name: ""
  type: "FC"
  arg {
    name: "use_cudnn"
    i: 1
  }
  arg {
    name: "cudnn_exhaustive_search"
    i: 0
  }
  arg {
    name: "order"
    s: "NCHW"
  }
  arg {
    name: "axis"
    i: 2
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/i2h"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/initial_hidden_state"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/initial_cell_state"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_w"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_b"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Copy/seq_lengths_copy"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_all"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_last"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state_all"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state_last"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/step_workspaces"
  name: ""
  type: "RecurrentNetwork"
  arg {
    name: "alias_dst"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_all"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_last"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state_all"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state_last"
  }
  arg {
    name: "link_internal"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/input_t"
  }
  arg {
    name: "backward_step_net"
    n {
      name: "RecurrentBackwardStep"
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Copy/seq_lengths_copy"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/timestep"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_grad"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_grad"
        name: ""
        type: "LSTMUnitGradient"
        arg {
          name: "sequence_lengths"
          i: 1
        }
        arg {
          name: "drop_states"
          i: 0
        }
        arg {
          name: "forget_bias"
          f: 0.0
        }
        is_gradient_op: true
      }
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_w"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_w_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_b_grad"
        output: "_dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_grad_autosplit_0"
        name: ""
        type: "FCGradient"
        arg {
          name: "use_cudnn"
          i: 1
        }
        arg {
          name: "cudnn_exhaustive_search"
          i: 0
        }
        arg {
          name: "order"
          s: "NCHW"
        }
        arg {
          name: "axis"
          i: 2
        }
        is_gradient_op: true
      }
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_grad"
        input: "_dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_grad_autosplit_0"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_grad"
        name: ""
        type: "Sum"
      }
      type: "simple"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_grad"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state_grad"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/input_t"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/timestep"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_w"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_b"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Copy/seq_lengths_copy"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state"
    }
  }
  arg {
    name: "backward_link_offset"
    ints: 1
    ints: 0
    ints: 1
    ints: 0
    ints: 0
  }
  arg {
    name: "timestep"
    s: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/timestep"
  }
  arg {
    name: "backward_link_external"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_states_grad"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_states_grad"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_states_grad"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_states_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/i2h_grad"
  }
  arg {
    name: "link_external"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_states"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/i2h"
  }
  arg {
    name: "outputs_with_grads"
    ints: 1
  }
  arg {
    name: "alias_offset"
    ints: 1
    ints: -1
    ints: 1
    ints: -1
  }
  arg {
    name: "link_offset"
    ints: 0
    ints: 1
    ints: 0
    ints: 1
    ints: 0
  }
  arg {
    name: "recompute_blobs_on_backward"
  }
  arg {
    name: "param_grads"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_w_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_b_grad"
  }
  arg {
    name: "backward_link_internal"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_grad"
  }
  arg {
    name: "param"
    ints: 3
    ints: 4
  }
  arg {
    name: "step_net"
    n {
      name: "lstm"
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_w"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_b"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t"
        name: ""
        type: "FC"
        arg {
          name: "use_cudnn"
          i: 1
        }
        arg {
          name: "cudnn_exhaustive_search"
          i: 0
        }
        arg {
          name: "order"
          s: "NCHW"
        }
        arg {
          name: "axis"
          i: 2
        }
      }
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/input_t"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t"
        name: ""
        type: "Sum"
        arg {
          name: "use_cudnn"
          i: 1
        }
        arg {
          name: "order"
          s: "NCHW"
        }
        arg {
          name: "cudnn_exhaustive_search"
          i: 0
        }
      }
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Copy/seq_lengths_copy"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/timestep"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state"
        name: ""
        type: "LSTMUnit"
        arg {
          name: "sequence_lengths"
          i: 1
        }
        arg {
          name: "drop_states"
          i: 0
        }
        arg {
          name: "forget_bias"
          f: 0.0
        }
      }
      type: "simple"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/input_t"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/timestep"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_w"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_b"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Copy/seq_lengths_copy"
      external_output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state"
      external_output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_state"
    }
  }
  arg {
    name: "recurrent_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_states"
  }
  arg {
    name: "enable_rnn_executor"
    i: 1
  }
  arg {
    name: "alias_src"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/cell_t_prev_states"
  }
  arg {
    name: "initial_recurrent_state_ids"
    ints: 1
    ints: 2
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_all"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/RemovePadding/hidden_output_all"
  name: ""
  type: "RemovePadding"
  arg {
    name: "padding_width"
    i: 0
  }
  arg {
    name: "end_padding_width"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/hidden_state_last"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Reshape/hidden_output_reshape"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Reshape/old_shape"
  name: ""
  type: "Reshape"
  arg {
    name: "shape"
    ints: -1
    ints: 32
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/output"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/PackRNNSequence/lstm_input"
  name: ""
  type: "PackRNNSequence"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/PackRNNSequence/lstm_input"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/AddPadding/rnn_input_pad"
  name: ""
  type: "AddPadding"
  arg {
    name: "padding_width"
    i: 0
  }
  arg {
    name: "end_padding_width"
    i: 1
  }
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Copy/seq_lengths_copy"
  name: ""
  type: "Copy"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Copy/seq_lengths_copy"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/ExpandDims/expanded_seq_lengths"
  name: ""
  type: "ExpandDims"
  arg {
    name: "dims"
    ints: 0
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/AddPadding/rnn_input_pad"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/i2h_w"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/i2h_b"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/i2h"
  name: ""
  type: "FC"
  arg {
    name: "use_cudnn"
    i: 1
  }
  arg {
    name: "cudnn_exhaustive_search"
    i: 0
  }
  arg {
    name: "order"
    s: "NCHW"
  }
  arg {
    name: "axis"
    i: 2
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/i2h"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/initial_hidden_state"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/initial_cell_state"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_w"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_b"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Copy/seq_lengths_copy"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_all"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_last"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state_all"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state_last"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/step_workspaces"
  name: ""
  type: "RecurrentNetwork"
  arg {
    name: "alias_dst"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_all"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_last"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state_all"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state_last"
  }
  arg {
    name: "link_internal"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/input_t"
  }
  arg {
    name: "backward_step_net"
    n {
      name: "RecurrentBackwardStep_3"
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Copy/seq_lengths_copy"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/timestep"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_grad"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_grad"
        name: ""
        type: "LSTMUnitGradient"
        arg {
          name: "sequence_lengths"
          i: 1
        }
        arg {
          name: "drop_states"
          i: 0
        }
        arg {
          name: "forget_bias"
          f: 0.0
        }
        is_gradient_op: true
      }
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_w"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_w_grad"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_b_grad"
        output: "_dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_grad_autosplit_0"
        name: ""
        type: "FCGradient"
        arg {
          name: "use_cudnn"
          i: 1
        }
        arg {
          name: "cudnn_exhaustive_search"
          i: 0
        }
        arg {
          name: "order"
          s: "NCHW"
        }
        arg {
          name: "axis"
          i: 2
        }
        is_gradient_op: true
      }
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_grad"
        input: "_dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_grad_autosplit_0"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_grad"
        name: ""
        type: "Sum"
      }
      type: "simple"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_grad"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state_grad"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/input_t"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/timestep"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_w"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_b"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Copy/seq_lengths_copy"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state"
    }
  }
  arg {
    name: "backward_link_offset"
    ints: 1
    ints: 0
    ints: 1
    ints: 0
    ints: 0
  }
  arg {
    name: "timestep"
    s: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/timestep"
  }
  arg {
    name: "backward_link_external"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_states_grad"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_states_grad"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_states_grad"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_states_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/i2h_grad"
  }
  arg {
    name: "link_external"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_states"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/i2h"
  }
  arg {
    name: "outputs_with_grads"
    ints: 1
  }
  arg {
    name: "alias_offset"
    ints: 1
    ints: -1
    ints: 1
    ints: -1
  }
  arg {
    name: "link_offset"
    ints: 0
    ints: 1
    ints: 0
    ints: 1
    ints: 0
  }
  arg {
    name: "recompute_blobs_on_backward"
  }
  arg {
    name: "param_grads"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_w_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_b_grad"
  }
  arg {
    name: "backward_link_internal"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_grad"
    strings: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_grad"
  }
  arg {
    name: "param"
    ints: 3
    ints: 4
  }
  arg {
    name: "step_net"
    n {
      name: "lstm_4"
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_w"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_b"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t"
        name: ""
        type: "FC"
        arg {
          name: "use_cudnn"
          i: 1
        }
        arg {
          name: "cudnn_exhaustive_search"
          i: 0
        }
        arg {
          name: "order"
          s: "NCHW"
        }
        arg {
          name: "axis"
          i: 2
        }
      }
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/input_t"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t"
        name: ""
        type: "Sum"
        arg {
          name: "use_cudnn"
          i: 1
        }
        arg {
          name: "order"
          s: "NCHW"
        }
        arg {
          name: "cudnn_exhaustive_search"
          i: 0
        }
      }
      op {
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Copy/seq_lengths_copy"
        input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/timestep"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state"
        output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state"
        name: ""
        type: "LSTMUnit"
        arg {
          name: "sequence_lengths"
          i: 1
        }
        arg {
          name: "drop_states"
          i: 0
        }
        arg {
          name: "forget_bias"
          f: 0.0
        }
      }
      type: "simple"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/input_t"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/timestep"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_w"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_b"
      external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Copy/seq_lengths_copy"
      external_output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state"
      external_output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_state"
    }
  }
  arg {
    name: "recurrent_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_states"
  }
  arg {
    name: "enable_rnn_executor"
    i: 1
  }
  arg {
    name: "alias_src"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_states"
    strings: "lstm/dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/cell_t_prev_states"
  }
  arg {
    name: "initial_recurrent_state_ids"
    ints: 1
    ints: 2
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_all"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/RemovePadding/hidden_output_all"
  name: ""
  type: "RemovePadding"
  arg {
    name: "padding_width"
    i: 0
  }
  arg {
    name: "end_padding_width"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/hidden_state_last"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Reshape/hidden_output_reshape"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Reshape/old_shape"
  name: ""
  type: "Reshape"
  arg {
    name: "shape"
    ints: -1
    ints: 32
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:values"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/sparse_lookup/output"
  name: ""
  type: "Gather"
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/sparse_lookup/output"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/pack_segments/batch_embedding"
  name: ""
  type: "PackSegments"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/pack_segments/batch_embedding"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/expand_dims/batch_embedding"
  name: ""
  type: "ExpandDims"
  arg {
    name: "dims"
    ints: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/expand_dims/batch_embedding"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/conv_kernel"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/conv_bias"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/output"
  name: ""
  type: "Conv"
  arg {
    name: "kernel_w"
    i: 3
  }
  arg {
    name: "pad_l"
    i: 0
  }
  arg {
    name: "pad_b"
    i: 0
  }
  arg {
    name: "stride_h"
    i: 1
  }
  arg {
    name: "stride_w"
    i: 1
  }
  arg {
    name: "pad_r"
    i: 2
  }
  arg {
    name: "order"
    s: "NHWC"
  }
  arg {
    name: "kernel_h"
    i: 1
  }
  arg {
    name: "pad_t"
    i: 0
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/output"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/squeeze/conv_output_squeeze"
  name: ""
  type: "Squeeze"
  arg {
    name: "dims"
    ints: 1
  }
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/squeeze/conv_output_squeeze"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/unpack_segments/conv_output_unpack"
  name: ""
  type: "UnpackSegments"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/unpack_segments/conv_output_unpack"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer/Sum_reducer"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/expand_dims/batch_embedding"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/conv_kernel"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/conv_bias"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/output"
  name: ""
  type: "Conv"
  arg {
    name: "kernel_w"
    i: 2
  }
  arg {
    name: "pad_l"
    i: 0
  }
  arg {
    name: "pad_b"
    i: 0
  }
  arg {
    name: "stride_h"
    i: 1
  }
  arg {
    name: "stride_w"
    i: 1
  }
  arg {
    name: "pad_r"
    i: 1
  }
  arg {
    name: "order"
    s: "NHWC"
  }
  arg {
    name: "kernel_h"
    i: 1
  }
  arg {
    name: "pad_t"
    i: 0
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/output"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/squeeze_auto_0/conv_output_squeeze"
  name: ""
  type: "Squeeze"
  arg {
    name: "dims"
    ints: 1
  }
}
op {
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/squeeze_auto_0/conv_output_squeeze"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/unpack_segments_auto_0/conv_output_unpack"
  name: ""
  type: "UnpackSegments"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/unpack_segments_auto_0/conv_output_unpack"
  input: "feature_preproc/output_features:SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS:lengths"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_auto_0/Sum_reducer"
  name: ""
  type: "LengthsSum"
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_auto_0/Sum_reducer"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output_auto_0"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output_auto_0_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/multiple_reducer_output_auto_0"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/concat/output"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/concat/output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "add_axis"
    i: 0
  }
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/concat/output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/w"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/b"
  output: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  input: "feature_preproc/output_features:SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  output: "dot/SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/output"
  name: ""
  type: "SparseLengthsSum"
}
op {
  input: "dot/SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/output"
  output: "dot/SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_0"
  output: "dot/SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_1"
  name: ""
  type: "Split"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  input: "feature_preproc/output_features:SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  output: "dot/SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/output"
  name: ""
  type: "SparseLengthsSum"
}
op {
  input: "dot/SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/output"
  output: "dot/SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_0"
  output: "dot/SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_1"
  name: ""
  type: "Split"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/w"
  input: "feature_preproc/output_features:SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM:values"
  input: "feature_preproc/output_features:SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM:lengths"
  output: "dot/SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/output"
  name: ""
  type: "SparseLengthsSum"
}
op {
  input: "dot/SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/output"
  output: "dot/SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_0"
  output: "dot/SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_1"
  name: ""
  type: "Split"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "nested/dense/Relu/relu"
  input: "dot/embedding_0/fc/w"
  input: "dot/embedding_0/fc/b"
  output: "dot/embedding_0/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/output"
  input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/fc/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/fc/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/fc/output"
  input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/output"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/Reshape/hidden_output_reshape"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/Reshape/hidden_output_reshape"
  input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/output"
  input: "dot/SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_0"
  input: "dot/SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_1"
  input: "dot/SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_0"
  input: "dot/SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_1"
  input: "dot/SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_0"
  input: "dot/SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM/split/output_1"
  input: "dot/embedding_0/fc/output"
  output: "dot/concat/output"
  output: "dot/concat/output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "add_axis"
    i: 1
  }
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/concat/output"
  input: "dot/concat/output"
  output: "dot/pairwise_similarity/dot/concat/output_matmul_auto_0"
  name: ""
  type: "BatchMatMul"
  arg {
    name: "trans_b"
    i: 1
  }
}
op {
  input: "dot/pairwise_similarity/dot/concat/output_matmul_auto_0"
  output: "dot/pairwise_similarity/dot/concat/output_matmul_auto_0_flatten"
  name: ""
  type: "Flatten"
}
op {
  input: "dot/pairwise_similarity/dot/concat/output_matmul_auto_0_flatten"
  input: "dot/pairwise_dot_product_gather_auto_0"
  output: "dot/pairwise_similarity/output"
  name: ""
  type: "BatchGather"
}
op {
  input: "dot/pairwise_similarity/output"
  input: "nested/dense/Relu/relu"
  output: "dot/concat_auto_0/output"
  output: "dot/concat_auto_0/output_concat_dims"
  name: ""
  type: "Concat"
  arg {
    name: "add_axis"
    i: 0
  }
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "dot/concat_auto_0/output"
  input: "over/fc/w"
  input: "over/fc/b"
  output: "over/fc/output"
  name: ""
  type: "FC"
}
op {
  input: "over/fc/output"
  output: "over/Relu/relu"
  name: ""
  type: "Relu"
}
op {
  input: "over/Relu/relu"
  input: "over/fc_auto_0/w"
  input: "over/fc_auto_0/b"
  output: "over/fc_auto_0/output"
  name: ""
  type: "FC"
}
op {
  input: "over/fc_auto_0/output"
  output: "over/Relu_auto_0/relu"
  name: ""
  type: "Relu"
}
op {
  input: "over/Relu_auto_0/relu"
  input: "fc/w"
  input: "fc/b"
  output: "fc/output"
  name: ""
  type: "FC"
}
op {
  input: "fc/output"
  output: "Sigmoid/sigmoid"
  name: ""
  type: "Sigmoid"
}
op {
  input: "supervision_preproc/Cast/label_float"
  output: "batch_lr_loss/label_float32"
  name: ""
  type: "Cast"
  arg {
    name: "to"
    i: 1
  }
}
op {
  input: "batch_lr_loss/label_float32"
  output: "batch_lr_loss/expanded_label"
  name: ""
  type: "ExpandDims"
  arg {
    name: "dims"
    ints: 1
  }
}
op {
  input: "fc/output"
  input: "batch_lr_loss/expanded_label"
  output: "batch_lr_loss/cross_entropy"
  name: ""
  type: "SigmoidCrossEntropyWithLogits"
  arg {
    name: "log_D_trick"
    i: 0
  }
  arg {
    name: "unjoined_lr_loss"
    i: 0
  }
}
op {
  input: "batch_lr_loss/cross_entropy"
  output: "batch_lr_loss/output"
  name: ""
  type: "AveragedLoss"
}
op {
  input: "batch_lr_loss/output"
  name: ""
  type: "EnforceFinite"
}
op {
  input: "supervision_preproc/Cast/label_float"
  output: "calibration/train_net/Cast"
  name: ""
  type: "Cast"
  arg {
    name: "to"
    i: 2
  }
}
op {
  input: "weight"
  output: "calibration/train_net/Cast_2"
  name: ""
  type: "Cast"
  arg {
    name: "to"
    i: 1
  }
}
op {
  input: "calibration/beta"
  input: "calibration/gamma"
  input: "calibration/train_net/Cast"
  input: "calibration/train_net/Cast_2"
  output: "calibration/beta"
  output: "calibration/gamma"
  name: ""
  type: "PriorCorrectionCalibrationAccumulate"
}
op {
  input: "calibration/beta"
  input: "calibration/gamma"
  input: "Sigmoid/sigmoid"
  output: "calibration/prediction"
  name: ""
  type: "PriorCorrectionCalibrationPrediction"
}
external_input: "float_features:values:keys"
external_input: "float_features:values:values"
external_input: "ZERO"
external_input: "float_features:lengths"
external_input: "id_list_features:values:values:lengths"
external_input: "id_list_features:values:keys"
external_input: "ZERO_RANGE"
external_input: "id_list_features:lengths"
external_input: "id_list_features:values:values:values"
external_input: "feature_preproc/feature_proc_group_0_starts"
external_input: "feature_preproc/feature_proc_group_0_ends"
external_input: "feature_preproc/feature_proc_group_1_starts"
external_input: "feature_preproc/feature_proc_group_1_ends"
external_input: "feature_preproc/feature_proc_group_2_starts"
external_input: "feature_preproc/feature_proc_group_2_ends"
external_input: "feature_preproc/feature_proc_group_2_one_hot_lens"
external_input: "feature_preproc/feature_proc_group_2_one_hot_vals"
external_input: "feature_preproc/feature_proc_shift_blob"
external_input: "feature_preproc/feature_proc_scale_blob"
external_input: "feature_preproc/sigrid_transform_transform_instance"
external_input: "feature_preproc/sigrid_transform_auto_0_transform_instance"
external_input: "label"
external_input: "nested/dense/fc/w"
external_input: "nested/dense/fc/b"
external_input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/w"
external_input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/w"
external_input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/b"
external_input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/w"
external_input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/w"
external_input: "dot/SPARSE_USER_COEFFICIENT_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/b"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/b"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/b"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/fc/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/fc/b"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/fc/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/fc/b"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/fc/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_2/fc/b"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/conv_kernel"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/conv_bias"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/conv_kernel"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/conv_bias"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/w"
external_input: "dot/SPARSE_AD_CROW_BODY_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/b"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_0/fc/b"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_0/Repeat_1/fc/b"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/i2h_w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/i2h_b"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/initial_hidden_state"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/initial_cell_state"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_0/lstm/gates_t_b"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/i2h_w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/i2h_b"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/initial_hidden_state"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/initial_cell_state"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_1/Repeat_1/lstm/gates_t_b"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/sparse_lookup/w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/conv_kernel"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv/conv_bias"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/conv_kernel"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/conv_auto_0/conv_bias"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/w"
external_input: "dot/SPARSE_AD_CROW_TITLE_HASHES_AUTO_FIRST_X_AUTO_UNIGRAM_AUTO_ADD_BIAS/Pool_Option_2/Repeat_0/fc/b"
external_input: "dot/SPARSE_USER_ENGAGED_PAGE_ID_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/w"
external_input: "dot/SPARSE_USER_CLK_AD_CAMPAIGN_IDS_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/w"
external_input: "dot/SPARSE_AD_OBJ_ID_AUTO_FIRST_X_AUTO_UNIGRAM/sparse_lookup/w"
external_input: "dot/embedding_0/fc/w"
external_input: "dot/embedding_0/fc/b"
external_input: "dot/pairwise_dot_product_gather_auto_0"
external_input: "over/fc/w"
external_input: "over/fc/b"
external_input: "over/fc_auto_0/w"
external_input: "over/fc_auto_0/b"
external_input: "fc/w"
external_input: "fc/b"
external_input: "weight"
external_input: "calibration/beta"
external_input: "calibration/gamma"
external_output: "calibration/prediction"
external_output: "supervision_preproc/Cast/label_float"
external_output: "weight"
external_output: "batch_lr_loss/output"
external_output: "Sigmoid/sigmoid"

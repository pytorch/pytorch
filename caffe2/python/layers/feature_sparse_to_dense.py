# @package sparse_to_dense
# Module caffe2.python.layers.sparse_to_dense


from collections import defaultdict

import numpy as np
from caffe2.python import schema
from caffe2.python.layers.layers import AccessedFeatures, ModelLayer


class FeatureSparseToDense(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        input_specs,
        name="feature_sparse_to_dense",
        default_dense_value=None,
        **kwargs
    ):
        """
        `input_specs` follows the format of FeatureSpec from schema. To be more
        precise it's a namedtuple that should have:
            'feature_type', 'feature_names', 'feature_ids'
        Default_dense_value can only be 0.0 or float("NaN"). Any input that isn't
        None will be NaN.
        """
        super().__init__(model, name, input_record, **kwargs)
        if default_dense_value is None:
            default_dense_value = 0.0
        default_dense_value = float(default_dense_value)
        assert (
            np.isnan(default_dense_value) or default_dense_value == 0.0
        ), "default_dense_value can only be 0.0 or NaN"

        self.input_specs = input_specs
        self.default_float_value = (
            model.global_constants["NAN"]
            if np.isnan(default_dense_value)
            else model.global_constants["ZERO"]
        )
        self.zero_range = model.global_constants["ZERO_RANGE"]

        outputs = []
        for field, feature_specs in self.input_specs:
            assert len(feature_specs.feature_names) == len(feature_specs.feature_ids)
            if feature_specs.feature_type == "FLOAT":
                outputs.append(
                    (
                        field,
                        schema.Scalar(
                            (np.float32, (len(feature_specs.feature_ids),)),
                            self.get_next_blob_reference(field + "_output"),
                        ),
                    )
                )
            elif feature_specs.feature_type == "ID_LIST":
                outputs.append(
                    (
                        field,
                        schema.Struct(
                            (
                                "ranges",
                                schema.Scalar(
                                    (np.int32, (len(feature_specs.feature_ids), 2)),
                                    self.get_next_blob_reference(field + "_ranges"),
                                ),
                            ),
                            (
                                "values",
                                schema.Scalar(
                                    np.int64,
                                    self.get_next_blob_reference(field + "_values"),
                                ),
                            ),
                        ),
                    )
                )
            elif feature_specs.feature_type == "ID_SCORE_LIST":
                outputs.append(
                    (
                        field,
                        schema.Struct(
                            (
                                "ranges",
                                schema.Scalar(
                                    (np.int32, (len(feature_specs.feature_ids), 2)),
                                    self.get_next_blob_reference(field + "_ranges"),
                                ),
                            ),
                            (
                                "ids",
                                schema.Scalar(
                                    np.int64,
                                    self.get_next_blob_reference(field + "_ids"),
                                ),
                            ),
                            (
                                "scores",
                                schema.Scalar(
                                    np.float32,
                                    self.get_next_blob_reference(field + "_scores"),
                                ),
                            ),
                        ),
                    )
                )
            elif feature_specs.feature_type == "EMBEDDING":
                # We don't know dimensions of embeddings in input data.
                # Even though they should match dimensions from feature config,
                # we keep ranges blob to check input data later.
                outputs.append(
                    (
                        field,
                        schema.Struct(
                            (
                                "ranges",
                                schema.Scalar(
                                    (np.int32, (len(feature_specs.feature_ids), 2)),
                                    self.get_next_blob_reference(field + "_ranges"),
                                ),
                            ),
                            (
                                "values",
                                schema.Scalar(
                                    np.float32,
                                    self.get_next_blob_reference(field + "_values"),
                                ),
                            ),
                        ),
                    )
                )
            elif feature_specs.feature_type == "GENERIC_FEATURE":
                # We don't know dimensions of embeddings in input data.
                # Even though they should match dimensions from feature config,
                # we keep ranges blob to check input data later.
                # Currently this schema with ranges and values is only for
                # generic type enum 1. If new types are implemented, we need to
                # modify the ParseGeneric operator, and this part accordingly
                outputs.append(
                    (
                        field,
                        schema.Struct(
                            (
                                "ranges",
                                schema.Scalar(
                                    (np.int32, (len(feature_specs.feature_ids), 2)),
                                    self.get_next_blob_reference(field + "_ranges"),
                                ),
                            ),
                            (
                                "values",
                                schema.Scalar(
                                    np.float32,
                                    self.get_next_blob_reference(field + "_values"),
                                ),
                            ),
                        ),
                    )
                )
            else:
                raise TypeError(
                    "Unsupported input type: {0}".format(feature_specs.feature_type)
                )

        # TODO(amalevich): This schema is producing ranges. And thus if there is
        # something using it it should support ranges as well. It might be
        # confusing, if we don't add better support for ranges/have it as a
        # first layer
        self.output_schema = schema.Struct(*outputs)

        # TODO(amalevich): Consider moving this data to schema, instead
        # Structs doesn't support attaching metadata to them and clonning
        # will break things badly, but this is the most elegant way to pass
        # this info around. Should we change it or it'll be too much work and
        # not worse it?
        for field, feature_specs in input_specs:
            schema.attach_metadata_to_scalars(
                self.output_schema[field], schema.Metadata(feature_specs=feature_specs)
            )

    # Add operators to all types that need to be densified
    def add_ops(self, net):
        record = self.input_record
        for field, feature_specs in self.input_specs:
            if feature_specs.feature_type == "FLOAT":
                net.SparseToDenseMask(
                    [
                        record[field].keys(),
                        record[field].values(),
                        self.default_float_value,
                        record[field].lengths(),
                    ],
                    [self.output_schema[field]()],
                    mask=feature_specs.feature_ids,
                )
            elif feature_specs.feature_type == "ID_LIST":
                id_list_ranges = net.LengthsToRanges(
                    record[field].values.lengths(), net.NextScopedBlob("id_list_ranges")
                )
                net.SparseToDenseMask(
                    [
                        record[field].keys(),
                        id_list_ranges,
                        self.zero_range,
                        record[field].lengths(),
                    ],
                    self.output_schema[field].ranges(),
                    mask=feature_specs.feature_ids,
                )
                # Alias helps to enforce the fact that all SparseToDense calls
                # produce new blobs.
                # Reusing blob names might result in some weird consequences
                # during the delivery time, when content of the blobs is
                # generated based on the inputSpecs.
                net.Alias(
                    record[field].values.items(), self.output_schema[field].values()
                )
            elif feature_specs.feature_type == "ID_SCORE_LIST":
                # TODO: merge this to the case above?
                id_list_ranges = net.LengthsToRanges(
                    record[field].values.lengths(),
                    net.NextScopedBlob("id_score_list_ranges"),
                )
                net.SparseToDenseMask(
                    [
                        record[field].keys(),
                        id_list_ranges,
                        self.zero_range,
                        record[field].lengths(),
                    ],
                    self.output_schema[field].ranges(),
                    mask=feature_specs.feature_ids,
                )
                # Alias helps to enforce the fact that all SparseToDense calls
                # produce new blobs.
                # Reusing blob names might result in some weird consequences
                # during the delivery time, when content of the blobs is
                # generated based on the inputSpecs.
                net.Alias(record[field].values.keys(), self.output_schema[field].ids())
                net.Alias(
                    record[field].values.values(), self.output_schema[field].scores()
                )
            elif feature_specs.feature_type == "EMBEDDING":
                ranges = net.LengthsToRanges(
                    record[field].values.lengths(),
                    net.NextScopedBlob("embeddings_ranges"),
                )
                net.SparseToDenseMask(
                    [
                        record[field].keys(),
                        ranges,
                        self.zero_range,
                        record[field].lengths(),
                    ],
                    self.output_schema[field].ranges(),
                    mask=feature_specs.feature_ids,
                )
                # Alias helps to enforce the fact that all SparseToDense calls
                # produce new blobs.
                # Reusing blob names might result in some weird consequences
                # during the delivery time, when content of the blobs is
                # generated based on the inputSpecs.
                net.Alias(
                    record[field].values.items(), self.output_schema[field].values()
                )
            elif feature_specs.feature_type == "GENERIC_FEATURE":
                (
                    feature_lengths_blob,
                    feature_ids_blob,
                    value_lengths_blob,
                    value_values_blob,
                ) = net.ParseGeneric(
                    [record[field]()],
                    ["feature_lengths", "feature_ids", "value_lengths", "value_values"],
                    feature_type_enum=1,
                )
                # Currently our implementation only supports
                # generic type enum 1. If new types are implemented, we need to
                # modify the ParseGeneric operator, the schema above,
                # and this part accordingly to parse the generic feature strings
                # into input_record

                ranges = net.LengthsToRanges(
                    value_lengths_blob, net.NextScopedBlob("generics_ranges")
                )
                net.SparseToDenseMask(
                    [feature_ids_blob, ranges, self.zero_range, feature_lengths_blob],
                    self.output_schema[field].ranges(),
                    mask=feature_specs.feature_ids,
                )
                # Alias helps to enforce the fact that all SparseToDense calls
                # produce new blobs.
                # Reusing blob names might result in some weird consequences
                # during the delivery time, when content of the blobs is
                # generated based on the inputSpecs.
                net.Alias(value_values_blob, self.output_schema[field].values())

    def get_metadata(self):
        metadata = []
        for field, feature_specs in self.input_specs:
            metadata.append(
                (
                    {
                        "type": feature_specs.feature_type,
                        "names": feature_specs.feature_names,
                        "ids": feature_specs.feature_ids,
                    },
                    self.output_schema[field].field_blobs(),
                    self.output_schema[field].field_types(),
                )
            )
            if feature_specs.feature_type == "FLOAT":
                metadata[-1][0]["cardinality"] = 1
        return metadata

    def get_accessed_features(self):
        accessed_features = defaultdict(list)

        # The features that are accessed are just those features that appear in
        # the input specs
        for field, feature_specs in self.input_specs:
            accessed_features[field].append(
                AccessedFeatures(
                    feature_specs.feature_type, set(feature_specs.feature_ids)
                )
            )

        return accessed_features

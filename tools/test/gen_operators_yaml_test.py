#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import unittest

from gen_operators_yaml import make_filter_from_options, verify_all_specified_present


class GenOperatorsYAMLTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_filter_creation(self):
        filter_func = make_filter_from_options(
            model_name="abc",
            model_versions=["100", "101"],
            model_assets=None,
            model_backends=None,
        )
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 102,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
            {
                "model": {
                    "name": "abcd",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
        ]

        filtered_configs = list(filter(filter_func, config))
        assert (
            len(filtered_configs) == 2
        ), "Expected 2 elements in filtered_configs, but got {}".format(
            len(filtered_configs)
        )

    def test_verification_success(self):
        filter_func = make_filter_from_options(
            model_name="abc",
            model_versions=["100", "101"],
            model_assets=["asset-1", "asset-2"],
            model_backends=None,
        )
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
        ]
        filtered_configs = list(filter(filter_func, config))
        try:
            verify_all_specified_present(
                model_assets=["asset-1", "asset-2"],
                model_versions=["100", "101"],
                selected_models_yaml=filtered_configs,
                rule_name="test",
                model_name="abc",
                new_style_rule=True,
            )
        except Exception:
            self.fail(
                "expected verify_all_specified_present to succeed instead it raised an exception"
            )

    def test_verification_fail(self):
        config = [
            {
                "model": {
                    "name": "abc",
                    "version": 100,
                    "asset": "asset-1",
                    "backend": "CPU",
                },
                "root_operators": [],
                "traced_operators": [],
            },
            {
                "model": {
                    "name": "abc",
                    "version": 101,
                    "asset": "asset-2",
                    "backend": "CPU",
                },
                "root_operators": [],
            },
        ]

        good_assets = ["asset-1", "asset-2"]
        good_versions = ["100", "101"]
        good_name = "abc"

        # Test bad asset
        filter_func_bad_asset = make_filter_from_options(
            model_name=good_name,
            model_versions=good_versions,
            model_assets=["asset-1", "asset-2", "asset-3"],
            model_backends=None,
        )
        filtered_configs_asset = list(filter(filter_func_bad_asset, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=["asset-1", "asset-2", "asset-3"],
                model_versions=good_versions,
                selected_models_yaml=filtered_configs_asset,
                rule_name="test",
                model_name=good_name,
                new_style_rule=True,
            )

        # Test bad version
        filter_func_bad_version = make_filter_from_options(
            model_name=good_name,
            model_versions=["100", "101", "102"],
            model_assets=good_assets,
            model_backends=None,
        )
        filtered_configs_version = list(filter(filter_func_bad_version, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=good_assets,
                model_versions=["100", "101", "102"],
                selected_models_yaml=filtered_configs_version,
                rule_name="test",
                model_name=good_name,
                new_style_rule=True,
            )

        # Test bad name
        filter_func_bad_name = make_filter_from_options(
            model_name="abcd",
            model_versions=good_versions,
            model_assets=good_assets,
            model_backends=None,
        )
        filtered_configs_name = list(filter(filter_func_bad_name, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=good_assets,
                model_versions=good_versions,
                selected_models_yaml=filtered_configs_name,
                rule_name="test",
                model_name="abcd",
                new_style_rule=True,
            )

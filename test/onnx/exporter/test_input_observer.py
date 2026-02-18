# Owner(s): ["module: onnx"]

import itertools

import pandas

import torch
from torch.onnx import InputObserver
from torch.onnx._internal.exporter._input_observer import _infer_dynamic_dimensions
from torch.testing._internal import common_utils


class TestInputObserver(common_utils.TestCase):
    def test_infer_dynamic_dimensions(self):
        self.assertEqual([2], _infer_dynamic_dimensions([(1, 2, 3), (1, 2, 4)]))
        self.assertEqual([0, 2], _infer_dynamic_dimensions([(1, 2, 3), (2, 2, 4)]))

    def test_io_captured_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())
        args = observer.infer_arguments()
        self.assertIsInstance(args, tuple)
        self.assertEqual(2, len(args))

    def test_io_captured_not_forward(self):
        class Model(torch.nn.Module):
            def notforward(self, w):
                return w.abs()

            def forward(self, x, y):
                return x + self.notforward(y)

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, method_name="notforward"):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({1: cst},), observer.infer_dynamic_shapes())
        args = observer.infer_arguments()
        self.assertIsInstance(args, tuple)
        self.assertEqual(1, len(args))

    def test_io_captured_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes()
        )
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))

    def test_io_captured_kwargs_bool(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, add=True):
                if add:
                    return x + y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6)), add=False),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7)), add=False),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)), add=False),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9)), add=False),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, add=None),
            observer.infer_dynamic_shapes(),
        )
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(3, len(args))

    def test_io_captured_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(z=torch.randn((5, 6)), w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((6, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((6, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((7, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((7, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((8, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((8, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(),
        )
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(4, len(args))

    def test_io_captured_optional_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x
                return x - y

        inputs = [
            (torch.randn((5, 6)),),
            (torch.randn((6, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((8, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())

    def test_infer_arguments_optional(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x
                return x - y

        inputs = [
            (torch.randn((5, 6)),),
            (torch.randn((6, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((8, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())
        infer_args = observer.infer_arguments(0)
        self.assertIsInstance(infer_args, tuple)
        self.assertEqual(len(infer_args), 2)
        self.assertIsInstance(infer_args[0], torch.Tensor)
        self.assertIsInstance(infer_args[1], torch.Tensor)
        self.assertEqual(infer_args[0].shape, (5, 6))
        self.assertEqual(infer_args[1].shape, (1, 0))

    def test_io_captured_optional_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(x=torch.randn((6, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((8, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes()
        )

    def test_io_captured_optional_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None, z=None, w=None):
                r = x + y if y is not None else x
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)),),
                dict(w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((6, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((6, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((7, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((7, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((8, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((8, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(),
        )

    def test_io_captured_not_supported_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if y is None:
                    return x
                if x is None:
                    return y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(y=torch.randn((1, 7))),
            dict(y=torch.randn((1, 7))),
            dict(y=torch.randn((1, 7))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaisesRegex(
            RuntimeError, "At least one call to the observed model"
        ):
            observer.infer_dynamic_shapes()

    def test_io_captured_incompatible_number_of_flattened_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if y is None:
                    return x
                if x is None:
                    return y[0]
                return x - y[0]

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(x=torch.randn((5, 7)), y=[torch.randn((1, 7))]),
            dict(x=torch.randn((5, 7)), y=[torch.randn((1, 7)), torch.randn((1, 7))]),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaisesRegex(RuntimeError, "Named argument 'y' has"):
            observer.infer_dynamic_shapes()

    def test_io_captured_incompatible_number_of_flattened_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x - y[0]

        inputs = [
            (torch.randn((5, 7)), [torch.randn((1, 7))]),
            (torch.randn((5, 7)), [torch.randn((1, 7)), torch.randn((1, 7))]),
        ]

        model = Model()
        observer = InputObserver()
        with self.assertRaisesRegex(RuntimeError, "No inputs were captured."):
            observer.infer_dynamic_shapes()
        with observer(model):
            for args in inputs:
                model(*args)
        with self.assertRaisesRegex(RuntimeError, "Positional argument 1 has"):
            observer.infer_dynamic_shapes()

    def test_io_captured_args_list(self):
        class Model(torch.nn.Module):
            def forward(self, x, y_list):
                return x + y_list[0] + y_list[1]

        inputs = [
            (torch.randn((5, 6)), [torch.randn((1, 6)), torch.randn((1, 6))]),
            (torch.randn((7, 7)), [torch.randn((1, 7)), torch.randn((1, 7))]),
            (torch.randn((7, 8)), [torch.randn((1, 8)), torch.randn((1, 8))]),
            (torch.randn((7, 9)), [torch.randn((1, 9)), torch.randn((1, 9))]),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            ({0: cst, 1: cst}, [{1: cst}, {1: cst}]), observer.infer_dynamic_shapes()
        )

    def test_io_captured_args_list_list(self):
        class Model(torch.nn.Module):
            def forward(self, x, y_list):
                return x + y_list[0] + y_list[1][0]

        inputs = [
            (torch.randn((5, 6)), [torch.randn((1, 6)), [torch.randn((1, 6))]]),
            (torch.randn((7, 7)), [torch.randn((1, 7)), [torch.randn((1, 7))]]),
            (torch.randn((7, 8)), [torch.randn((1, 8)), [torch.randn((1, 8))]]),
            (torch.randn((7, 9)), [torch.randn((1, 9)), [torch.randn((1, 9))]]),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            ({0: cst, 1: cst}, [{1: cst}, [{1: cst}]]), observer.infer_dynamic_shapes()
        )

    def test_io_captured_args_dict(self):
        class Model(torch.nn.Module):
            def forward(self, x, y_dict):
                return x + y_dict["x"] + y_dict["y"]

        inputs = [
            (torch.randn((5, 6)), dict(x=torch.randn((1, 6)), y=torch.randn((1, 6)))),
            (torch.randn((7, 7)), dict(x=torch.randn((1, 7)), y=torch.randn((1, 7)))),
            (torch.randn((7, 8)), dict(x=torch.randn((1, 8)), y=torch.randn((1, 8)))),
            (torch.randn((7, 9)), dict(x=torch.randn((1, 9)), y=torch.randn((1, 9)))),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = ({0: cst, 1: cst}, dict(x={1: cst}, y={1: cst}))
        model = Model()
        torch.export.export(model, inputs[-1], dynamic_shapes=expected)

        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)

        self.assertEqual(
            ({0: cst, 1: cst}, dict(x={1: cst}, y={1: cst})),
            observer.infer_dynamic_shapes(),
        )

    def test_io_captured_args_dict_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y_list, z_tuple=None):
                if z_tuple is None:
                    return x + y_list[0] + y_list[1]
                return x + y_list[0] + y_list[1] + z_tuple[0] + z_tuple[1]

        inputs = [
            ((torch.randn((5, 6)), [torch.randn((5, 6)), torch.randn((1, 6))]), {}),
            (
                (torch.randn((6, 7)), [torch.randn((6, 7)), torch.randn((1, 7))]),
                {"z_tuple": (torch.randn((6, 7)), torch.randn((1, 7)))},
            ),
            (
                (torch.randn((7, 8)), [torch.randn((7, 8)), torch.randn((1, 8))]),
                {"z_tuple": (torch.randn((7, 8)), torch.randn((1, 8)))},
            ),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = dict(
            x={0: cst, 1: cst},
            y_list=[{0: cst, 1: cst}, {1: cst}],
            z_tuple=({0: cst, 1: cst}, {1: cst}),
        )
        model = Model()
        torch.export.export(
            model, inputs[-1][0], kwargs=inputs[-1][1], dynamic_shapes=expected
        )

        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
        self.assertEqual(expected, observer.infer_dynamic_shapes())

    def test_io_captured_custom_class(self):
        class TestCustomClass:
            def __init__(self, keys, values):
                self.data = list(zip(keys, values))

        def _flatten(custom):
            data = custom.data
            flat = list(itertools.chain.from_iterable(data))
            keys = list(
                itertools.chain.from_iterable(
                    (f"key_{i}", f"value_{i}") for i in range(len(data))
                )
            )
            return flat, keys

        def _flatten_with_keys(custom):
            values, context = _flatten(custom)
            return [
                (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
            ], context

        def _unflatten(values, context, output_type=None):
            return TestCustomClass(values[::2], values[1::2])

        torch.utils._pytree.register_pytree_node(
            TestCustomClass,
            _flatten,
            _unflatten,
            serialized_type_name="onnxtest.TestCustomClass",
            flatten_with_keys_fn=_flatten_with_keys,
        )

        class Model(torch.nn.Module):
            def forward(self, x, custom=None):
                if not custom:
                    return x
                data = custom.data
                return x + data[0][0] + data[0][1] + data[1][0] + data[1][1]

        inputs = [
            (torch.randn((5, 6)),),
            (
                torch.randn((6, 7)),
                TestCustomClass(
                    [torch.randn((6, 7)), torch.randn((1, 7))],
                    [torch.randn((1, 7)), torch.randn((6, 7))],
                ),
            ),
            (
                torch.randn((7, 8)),
                TestCustomClass(
                    [torch.randn((7, 8)), torch.randn((1, 8))],
                    [torch.randn((1, 8)), torch.randn((7, 8))],
                ),
            ),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = (
            {0: cst, 1: cst},
            [{0: cst, 1: cst}, {1: cst}, {1: cst}, {0: cst, 1: cst}],
        )
        flat = torch.utils._pytree.tree_flatten(inputs[-1])[0]
        self.assertEqual(len(flat), 5)

        model = Model()
        model(*inputs[-1])
        torch.export.export(model, inputs[-1], dynamic_shapes=expected)
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(expected, observer.infer_dynamic_shapes())

    def test_io_captured_args_kwargs_dynamic_batch(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(z=torch.randn((5, 6)), w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((5, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={0, "z"}),
        )
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={"x", "z"}),
        )

    def test_io_captured_different_order(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(w=torch.randn((1, 6)), z=torch.randn((5, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(w=torch.randn((1, 8)), z=torch.randn((5, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={0, "z"}),
        )
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={"x", "z"}),
        )
        epo = torch.onnx.export(
            model,
            (),
            kwargs=observer.infer_arguments(),
            dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
        )
        data = observer.check_discrepancies(epo, progress_bar=False)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-5)

    def test_io_check_discrepancies(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)

        epo = torch.onnx.export(
            model,
            observer.infer_arguments(),
            dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
        )
        data = observer.check_discrepancies(epo, progress_bar=False)
        self.assertEqual(len(data), 3)
        self.assertIsInstance(data[0], dict)
        self.assertLess(max(obs["abs"] for obs in data), 1e-5)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-5)

    def test_io_infer_arguments(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(w=torch.randn((1, 6)), z=torch.randn((5, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(w=torch.randn((1, 8)), z=torch.randn((5, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        iargs = observer.infer_arguments(
            dict(w=torch.randn((1, 6)), z=torch.randn((5, 6)))
        )
        self.assertEqual(len(iargs), 4)
        self.assertEqual(iargs["x"].shape, (5, 0))
        self.assertEqual(iargs["y"].shape, (1, 0))
        self.assertEqual(iargs["w"].shape, (1, 6))
        self.assertEqual(iargs["z"].shape, (5, 6))

        iargs = observer.infer_arguments((torch.randn((5, 6)), torch.randn((1, 6))))
        self.assertEqual(len(iargs), 4)
        self.assertEqual(iargs["x"].shape, (5, 6))
        self.assertEqual(iargs["y"].shape, (1, 6))
        self.assertEqual(iargs["w"].shape, (1, 0))
        self.assertEqual(iargs["z"].shape, (5, 0))

    def test_io_mixed_args_kwargs_as_dict_1(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if y is None:
                    return x
                return x + y

        inputs = [
            ((torch.randn((5, 6)),), dict()),
            ((), dict(x=torch.randn((5, 7)), y=torch.randn((5, 7)))),
            ((torch.randn((5, 8)),), dict()),
            ((), dict(x=torch.randn((5, 9)), y=torch.randn((5, 9)))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 4)
        observer.infer_dynamic_shapes()
        for cand in observer.info.inputs:
            cand.str_obs()
            self.assertEqual(
                len(cand.flat_list),
                len([t for t in cand.aligned_flat_list if t is not None]),
            )

        cst = torch.export.Dim.DYNAMIC
        dynamic_shapes = observer.infer_dynamic_shapes()
        self.assertEqual({"x": {1: cst}, "y": {1: cst}}, dynamic_shapes)
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))
        self.assertEqual(len([v for v in args.values() if v is not None]), 2)

    def test_io_int_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None, option=1):
                if option == 1:
                    return x + y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 7)), y=torch.randn((5, 7)), option=0),
            dict(x=torch.randn((5, 9)), y=torch.randn((5, 9)), option=0),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for kwargs in inputs:
                model(**kwargs)
        kwargs = observer.infer_arguments()
        self.assertIn("option", kwargs)
        self.assertEqual(kwargs["option"], 0)
        shapes = observer.infer_dynamic_shapes()
        self.assertIn("option", shapes)
        self.assertEqual(shapes["option"], None)
        ep = torch.export.export(model, (), kwargs=kwargs, dynamic_shapes=shapes)
        torch.testing.assert_close(model(**kwargs), ep.module()(**kwargs))
        epo = torch.onnx.export(model, (), kwargs=kwargs, dynamic_shapes=shapes)
        proto = epo.model_proto
        self.assertEqual(["x", "y"], [i.name for i in proto.graph.input])

    def test_io_mixed_args_kwargs_as_dict_2(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if x is None:
                    return y
                return x + y

        inputs = [
            ((), dict(y=torch.randn((5, 6)))),
            ((torch.randn((5, 7)), torch.randn((5, 7))), dict()),
            ((), dict(y=torch.randn((5, 8)))),
            ((torch.randn((5, 9)), torch.randn((5, 9))), dict()),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 4)
        with self.assertRaises(RuntimeError):
            observer.infer_dynamic_shapes()


if __name__ == "__main__":
    common_utils.run_tests()

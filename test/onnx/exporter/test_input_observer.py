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

    def test_infer_dynamic_shapes_missing(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                token_type_ids=None,
                cache_position=None,
            ):
                return input_ids

        inputs = [
            dict(
                input_ids=torch.ones((1, 28), dtype=torch.int64),
                pixel_values=torch.ones((1, 3, 112, 112), dtype=torch.int64),
                attention_mask=torch.ones((1, 28), dtype=torch.int64),
                position_ids=torch.ones((1, 28), dtype=torch.int64),
                token_type_ids=torch.ones((1, 28), dtype=torch.int64),
                cache_position=torch.ones((28,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 29), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 28, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 30), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 29, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing=dict(pixel_values=torch.empty((0, 3, 112, 112)))
        )
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = {
            "input_ids": {0: cst, 1: cst},
            "pixel_values": {0: cst},
            "attention_mask": {0: cst, 1: cst},
            "position_ids": {0: cst, 1: cst},
            "past_key_values": {0: cst, 2: cst},
            "token_type_ids": {0: cst, 1: cst},
            "cache_position": {0: cst},
        }
        self.assertEqual(expected, shapes)
        kwargs = observer.infer_arguments()
        self.assertEqual(list(expected), list(kwargs))
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"].shape)

    def test_infer_dynamic_shapes_missing_args(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                past_key_values=None,
            ):
                return input_ids

        inputs = [
            (
                torch.ones((1, 28), dtype=torch.int64),
                torch.ones((1, 3, 112, 112), dtype=torch.int64),
                torch.ones((1, 28), dtype=torch.int64),
            ),
            (
                torch.ones((1, 1), dtype=torch.int64),
                None,
                torch.ones((1, 29), dtype=torch.int64),
                torch.rand((1, 1, 28, 32)),
            ),
            (
                torch.ones((1, 1), dtype=torch.int64),
                None,
                torch.ones((1, 30), dtype=torch.int64),
                torch.rand((1, 1, 29, 32)),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing={1: torch.empty((0, 3, 112, 112), dtype=torch.int64)}
        )
        with observer(model):
            for args in inputs:
                model(*args)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = ({0: cst, 1: cst}, {0: cst}, {0: cst, 1: cst}, {0: cst, 2: cst})
        self.assertEqual(expected, shapes)
        args = observer.infer_arguments()
        self.assertEqual(len(expected), len(args))
        self.assertEqual((0, 3, 112, 112), args[1].shape)

    def test_infer_dynamic_shapes_missing_kwargs_nested(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                token_type_ids=None,
                cache_position=None,
            ):
                return input_ids

        inputs = [
            dict(
                input_ids=torch.ones((1, 28), dtype=torch.int64),
                pixel_values=(
                    torch.ones((1, 3, 112, 112), dtype=torch.int64),
                    torch.ones((1, 3, 112, 112), dtype=torch.int64),
                ),
                attention_mask=torch.ones((1, 28), dtype=torch.int64),
                position_ids=torch.ones((1, 28), dtype=torch.int64),
                token_type_ids=torch.ones((1, 28), dtype=torch.int64),
                cache_position=torch.ones((28,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 29), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 28, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 30), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 29, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing=dict(
                pixel_values=(
                    torch.empty((0, 3, 112, 112), dtype=torch.int64),
                    torch.empty((0, 3, 112, 112), dtype=torch.int64),
                )
            )
        )
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = {
            "input_ids": {0: cst, 1: cst},
            "pixel_values": ({0: cst}, {0: cst}),
            "attention_mask": {0: cst, 1: cst},
            "position_ids": {0: cst, 1: cst},
            "past_key_values": {0: cst, 2: cst},
            "token_type_ids": {0: cst, 1: cst},
            "cache_position": {0: cst},
        }
        self.assertEqual(expected, shapes)
        kwargs = observer.infer_arguments()
        self.assertEqual(list(expected), list(kwargs))
        self.assertIsInstance(kwargs["pixel_values"], tuple)
        self.assertEqual(2, len(kwargs["pixel_values"]))
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"][0].shape)
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"][1].shape)

    def test_io_captured_kwargs_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, **kwargs):
                return x + kwargs["y"]

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
        ds = observer.infer_dynamic_shapes()
        self.assertEqual(dict(x={0: cst, 1: cst}, kwargs=dict(y={1: cst})), ds)
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))
        self.assertEqual(["x", "y"], list(args))

        dynamic_shapes = torch.export.AdditionalInputs()
        for kwargs in inputs:
            dynamic_shapes.add((), kwargs)
        dss = dynamic_shapes.dynamic_shapes(model, (), inputs[0])
        self.assertEqual({"x": (cst, cst), "kwargs": {"y": (None, cst)}}, dss)

    def test_io_captured_kwargs_kwargs_with_args(self):
        class Model(torch.nn.Module):
            def forward(self, a, *args, **kwargs):
                return a - args[0] * args[1] + kwargs["x"] - kwargs["y"]

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((5, 6)), torch.randn((5, 6))),
                dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            ),
            (
                (torch.randn((7, 7)), torch.randn((7, 7)), torch.randn((7, 7))),
                dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 2)
        for i in range(2):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        ds = observer.infer_dynamic_shapes()
        self.assertEqual(
            {
                "a": {0: cst, 1: cst},
                "args": ({0: cst, 1: cst}, {0: cst, 1: cst}),
                "kwargs": {"x": {0: cst, 1: cst}, "y": {1: cst}},
            },
            ds,
        )

        dynamic_shapes = torch.export.AdditionalInputs()
        for args, kwargs in inputs:
            dynamic_shapes.add(args, kwargs)
        dss = dynamic_shapes.dynamic_shapes(model, *inputs[0])
        self.assertEqual(
            {
                "a": (cst, cst),
                "args": ((cst, cst), (cst, cst)),
                "kwargs": {"x": (cst, cst), "y": (None, cst)},
            },
            dss,
        )

        with self.assertRaises(RuntimeError):
            observer.infer_arguments()

        args, kwargs = observer.infer_arguments(as_args_kwargs=True)
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(["x", "y"], list(kwargs))
        self.assertIsInstance(args, tuple)
        self.assertEqual(len(args), 3)


if __name__ == "__main__":
    common_utils.run_tests()

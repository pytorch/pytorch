# Owner(s): ["module: dynamo"]

# ruff: noqa: TRY002
# flake8: noqa

import dataclasses
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Optional, Tuple

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.nn
import torch.utils.checkpoint
from torch._dynamo.testing import same
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase


class SimpleDict(dict):
    pass


class DictTests(torch._dynamo.test_case.TestCase):
    def test_dict_subclass_instantiation(self):
        def fn(x):
            sd = SimpleDict(x=5)
            return sd["x"] * x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_local_mutation(self):
        def fn(x):
            sd = SimpleDict(x=5)
            z = sd["x"] * x
            sd["x"] = 10
            return z * sd["x"]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_local_with_non_dict_method(self):
        # Checks that add_1 method is inlined
        class MethodDict(dict):
            def add_1(self, x):
                return x + 1

        def fn(x):
            sd = MethodDict(x=5)
            z = sd["x"] * x
            sd["x"] = 10
            return sd.add_1(z * sd["x"])

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_methods_fallback_readonly(self):
        sd = SimpleDict()
        sd[2] = 5
        sd[4] = 10
        # check that regular attr accesses work well
        sd.attr = 4

        def fn(x):
            for value in sd.values():
                x = x * value
            for key in sd.keys():
                x = x * key
            for k, v in sd.items():
                x = x * k
                x = x * v
            # for k in sd:
            #     x = x * k

            if 1 in sd:
                x = x * 2
            else:
                x = x * 3

            x = x * sd.get(2, 0)
            x = x * sd.get(3, 4)
            x = len(sd) * x
            x = x * sd.attr
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure a recompilation
        sd[6] = 15
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_instantiation_return(self):
        def fn(x):
            sd = SimpleDict(x=5 * x)
            sd["y"] = 10
            return sd

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(type(ref), type(res))
        self.assertEqual(ref["x"], res["x"])
        self.assertEqual(ref["y"], res["y"])

    def test_dict_subclass_methods_fallback_mutation(self):
        def fn(sd, x):
            for value in sd.values():
                x = x * value
            sd[6] = 14
            for key in sd.keys():
                x = x * key
            for k, v in sd.items():
                x = x * k
                x = x * v
            # for k in sd:
            #     x = x * k

            if 1 in sd:
                x = x * 2
            else:
                x = x * 3

            x = x * sd.get(2, 0)
            x = x * sd.get(3, 4)
            x = len(sd) * x
            x = x * sd.attr
            sd.attr = 10
            x = x * sd.attr
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        sd1 = SimpleDict()
        sd1[2] = 5
        sd1[4] = 10
        sd1.attr = 4

        sd2 = SimpleDict()
        sd2[2] = 5
        sd2[4] = 10
        sd2.attr = 4
        self.assertTrue(sd1 == sd2)

        self.assertEqual(fn(sd1, x), opt_fn(sd2, x))
        self.assertTrue(sd1 == sd2)
        self.assertTrue(sd1.attr == sd2.attr)

    def test_dict_subclass_setitem(self):
        class SetItemDict(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value + 1)

        def fn(x):
            sd = SetItemDict(x=5 * x)
            sd["y"] = 10
            return sd

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(type(ref), type(res))
        self.assertEqual(ref["x"], res["x"])
        self.assertEqual(ref["y"], res["y"])


def is_tensor(x):
    import torch

    return isinstance(x, torch.Tensor)


class ModelOutput(OrderedDict):
    """
    Copied from transformers.
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Copied from transformers
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Copied from transformers
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class TestModelOutput(torch._dynamo.test_case.TestCase):
    def test_mo_create(self):
        def fn(a, b):
            tmp = BaseModelOutput(a + 1, attentions=b + 3)
            return tmp

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=2)

    def test_mo_assign(self):
        def fn(a, b):
            tmp = BaseModelOutput(last_hidden_state=b + 3)
            tmp.hidden_states = a + 7
            tmp["attentions"] = a + b + 6
            return tmp

        args = [torch.randn(10), torch.randn(10)]
        obj1 = fn(*args)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1.last_hidden_state, obj2.last_hidden_state))
        self.assertTrue(same(obj1.hidden_states, obj2.hidden_states))
        self.assertTrue(same(obj1.attentions, obj2.attentions))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def _common(self, fn, op_count):
        args = [
            BaseModelOutput(
                last_hidden_state=torch.randn(10), attentions=torch.randn(10)
            )
        ]
        obj1 = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, op_count)

    def test_mo_getattr(self):
        def fn(obj: BaseModelOutput):
            x = obj.last_hidden_state * 10
            if obj.hidden_states is not None:
                x += obj.hidden_states
            if obj.attentions is not None:
                x += obj.attentions
            return x

        self._common(fn, 2)

    def test_mo_getattr_missing(self):
        def fn(obj: BaseModelOutput):
            if getattr(obj, "asdf", None) is not None:
                obj.asdf += 1
            return obj.attentions + 1

        self._common(fn, 1)

    def test_mo_getitem(self):
        def fn(obj: BaseModelOutput):
            x = obj["last_hidden_state"] * 10
            if "hidden_stats" in obj:
                x += obj["hidden_states"]
            if "attentions" in obj:
                x += obj["attentions"]
            return x

        self._common(fn, 2)

    def test_mo_tuple(self):
        def fn(obj: BaseModelOutput):
            a, b = obj.to_tuple()
            return a + b * 10

        self._common(fn, 2)

    def test_mo_index(self):
        def fn(obj: BaseModelOutput):
            return obj[0] * 10 + obj[1]

        self._common(fn, 2)

    def test_mo_init(self):
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        def fn(obj):
            class_fields = dataclasses.fields(obj)
            assert len(class_fields)
            assert all(field.default is None for field in class_fields[1:])
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none

            total = getattr(obj, class_fields[0].name)
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        tensors = [torch.randn(10), torch.randn(10), torch.randn(10)]
        obj1 = MyDataClass(*tensors)
        correct1 = fn(obj1)

        obj2 = MyDataClass(*tensors)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj2), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_mo_init2(self):
        # this ModelOutput subclass runs a different __post_init__ codepath
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            x: torch.FloatTensor = None

        def fn(x):
            obj = MyDataClass(x=x * 5)
            return obj

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(inp).x, opt_fn(inp).x)

    def test_mo_init_with_disable(self):
        # Can result in "non-function or method super: <slot wrapper '__setattr__' of 'object' objects>"
        # graph breaks (although it may not be the first)
        # Minimal repro for https://github.com/pytorch/pytorch/issues/126028
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            x: torch.FloatTensor = None

        @torch._dynamo.disable(recursive=False)
        def fn(x):
            return MyDataClass(x=x)

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(inp).x, opt_fn(inp).x)

    def test_mo_newkey(self):
        obj = BaseModelOutput()

        def fn(obj):
            return obj["wwww"] + 1

        inp = torch.randn(3, 3)
        obj["wwww"] = inp
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    def test_mo_from_outside(self):
        def fn(obj):
            return obj.attentions + 1

        obj = BaseModelOutput(attentions=torch.randn(3, 3))
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    def test_mo_reconstruct_bytecode(self):
        def fn(inp):
            return BaseModelOutput(attentions=inp + 1)

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(inp).attentions, opt_fn(inp).attentions)

    def test_none(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                return CausalLMOutputWithPast(loss=None, logits=x)[0]

        model = Model()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(1, 1, 1, 1)

        self.assertTrue(same(model(x), opt_model(x)))

    def test_reconstruction(self):
        torch._export.utils.register_dataclass_as_pytree_node(
            CausalLMOutputWithPast,
            serialized_type_name="test_reconstruction_CausalLMOutputWithPast",
        )

        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                return CausalLMOutputWithPast(loss=x, logits=None)

        model = Model()
        x = torch.randn(1, 1, 1, 1)
        eo = torch._dynamo.export(Model(), aten_graph=True)(x)
        self.assertTrue(same(model(x), eo.graph_module(x)))


class TestModelOutputBert(TestCase):
    def test_HF_bert_model_output(self, device):
        class BertPooler(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense = torch.nn.Linear(768, 768).to(device)
                self.activation = torch.nn.Tanh()

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token.
                first_token_tensor = hidden_states[:, 0]
                pooled_output = self.dense(first_token_tensor)
                pooled_output = self.activation(pooled_output)
                return pooled_output

        class BertEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                hidden_states: torch.Tensor,
            ) -> BaseModelOutputWithPastAndCrossAttentions:
                return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    cross_attentions=None,
                )

        class BertModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = BertEncoder()
                self.pooler = BertPooler()

            def forward(
                self,
                sequence_output: torch.Tensor,
            ) -> BaseModelOutputWithPoolingAndCrossAttentions:
                encoder_outputs = self.encoder(sequence_output)
                # test __getitem__ and to_tuple
                sequence_output = encoder_outputs[0]
                pooled_output = (
                    self.pooler(sequence_output) if self.pooler is not None else None
                )
                # test CustomDictVariable.create
                result = BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
                )
                # test __setattr__
                result.pooler_output = pooled_output
                # test __setitem__
                result["pooler_output"] = pooled_output
                return result

        sequence_output = torch.rand(1, 12, 768).to(device)
        model = BertModel()
        orig_result = model(sequence_output)
        compiled_model = torch.compile(model, backend="eager")
        compiled_result = compiled_model(sequence_output)
        self.assertTrue(
            torch.allclose(
                orig_result.last_hidden_state, compiled_result.last_hidden_state
            )
        )
        self.assertTrue(
            torch.allclose(orig_result.pooler_output, compiled_result.pooler_output)
        )


devices = ["cpu"]

instantiate_device_type_tests(TestModelOutputBert, globals(), only_for=devices)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

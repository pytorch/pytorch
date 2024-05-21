# Owner(s): ["module: dynamo"]
import dataclasses
import unittest.mock

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same

try:
    from transformers import modeling_outputs
    from transformers.configuration_utils import PretrainedConfig
    from transformers.file_utils import ModelOutput
    from transformers.modeling_outputs import (
        BaseModelOutput,
        BaseModelOutputWithPastAndCrossAttentions,
        BaseModelOutputWithPoolingAndCrossAttentions,
        CausalLMOutputWithPast,
    )
except ImportError:
    modeling_outputs = None


def maybe_skip(fn):
    if modeling_outputs is None:
        return unittest.skip("requires HuggingFace")(fn)
    return fn


class TestHFPretrained(torch._dynamo.test_case.TestCase):
    @maybe_skip
    def test_pretrained(self):
        def fn(a, tmp):
            if hasattr(tmp, "somekey"):
                a = a + 1
            if tmp.return_dict:
                return a + torch.ones(2) * tmp.max_length
            return a

        x = torch.randn(2)
        tmp = PretrainedConfig(return_dict=True, max_length=20)
        ref = fn(x, tmp)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(x, tmp)
        self.assertTrue(same(ref, res))


class TestModelOutput(torch._dynamo.test_case.TestCase):
    @maybe_skip
    def test_mo_create(self):
        def fn(a, b):
            tmp = BaseModelOutput(a + 1, attentions=b + 3)
            return tmp

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=2)

    @maybe_skip
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

    @maybe_skip
    def test_mo_getattr(self):
        def fn(obj: BaseModelOutput):
            x = obj.last_hidden_state * 10
            if obj.hidden_states is not None:
                x += obj.hidden_states
            if obj.attentions is not None:
                x += obj.attentions
            return x

        self._common(fn, 2)

    @maybe_skip
    def test_mo_getitem(self):
        def fn(obj: BaseModelOutput):
            x = obj["last_hidden_state"] * 10
            if "hidden_stats" in obj:
                x += obj["hidden_states"]
            if "attentions" in obj:
                x += obj["attentions"]
            return x

        self._common(fn, 2)

    @maybe_skip
    def test_mo_tuple(self):
        def fn(obj: BaseModelOutput):
            a, b = obj.to_tuple()
            return a + b * 10

        self._common(fn, 2)

    @maybe_skip
    def test_mo_index(self):
        def fn(obj: BaseModelOutput):
            return obj[0] * 10 + obj[1]

        self._common(fn, 2)

    @maybe_skip
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
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        self.assertTrue(same(opt_fn(obj2), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    @maybe_skip
    def test_HF_bert_model_output(self):
        class BertPooler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(768, 768).to("cuda")
                self.activation = torch.nn.Tanh()

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token.
                first_token_tensor = hidden_states[:, 0]
                pooled_output = self.dense(first_token_tensor)
                pooled_output = self.activation(pooled_output)
                return pooled_output

        class BertEncoder(torch.nn.Module):
            def __init__(self):
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
            def __init__(self):
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

        sequence_output = torch.rand(1, 12, 768).to("cuda")
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

    @maybe_skip
    def test_none(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                return CausalLMOutputWithPast(loss=None, logits=x)[0]

        model = Model()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(1, 1, 1, 1)

        self.assertTrue(same(model(x), opt_model(x)))

    @maybe_skip
    def test_reconstruction(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                return CausalLMOutputWithPast(loss=x, logits=None)

        model = Model()
        x = torch.randn(1, 1, 1, 1)
        eo = torch._dynamo.export(Model(), aten_graph=True)(x)
        self.assertTrue(same(model(x), eo.graph_module(x)))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

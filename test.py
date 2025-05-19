import torch
import torch._dynamo.test_case
import torch._inductor.test_case

class ViewAndMutationMetaFromDynamo(torch._dynamo.test_case.TestCase):

    # Note: These 4 tests are to evalaute the feasability for the four 
    # fields  in metadata analysis that were identifed as diffifuclt to 
    # extract from dynamo

    # We want to run each one with TORCH_LOGS=dynamo, aot_autograd, aot_eager
    # To view the created artifact and verify we have sufficient info
    def test_output_alias_info_functional_tensor(self):
        @torch.compile
        def f(x):
            return x[1].view(-1)

        x = torch.randn(4, 4, requires_grad=True)
        out = f(x)

    def test_input_alias_info_mutations_hidden_from_autograd(self):
        # TODO: Need example of custom triton kernel here or HOPs
        # since these are likely the types of mutations that would be hidden from autograd
        pass
    
    def test_traced_tangents(self):
        # This one is likely fine, so we can most likely ignore but worth verifying 
        # just in case
        pass

    def test_tokens(self):
        # Map of effect type (ex. _EffectType.ORDERED) to token
        # FunctionalTensorMode would have populated this, so we need to validate 
        # that we can populate this from dynamo - should be similar to HOPs and Triton
        # kernels
        pass

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
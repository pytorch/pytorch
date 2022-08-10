# Owner(s): ["module: masked operators"]

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    make_tensor,
    instantiate_parametrized_tests,
)

from torch.testing._internal.common_methods_invocations import (
    SampleInput,
)

from torch.masked.maskedtensor.core import _masks_match, _tensors_match


def _compare_mt_t(mt_result, t_result):
    mask = mt_result.get_mask()
    mt_result_data = mt_result.get_data()
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()
    if mt_result_data.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_result_data = mt_result_data.to_dense()
    a = mt_result_data.detach().masked_fill_(~mask, 0)
    b = t_result.detach().masked_fill_(~mask, 0)
    if not _tensors_match(a, b, exact=False):
        raise ValueError("The data in MaskedTensor a and Tensor b do not match")

def _compare_mts(mt1, mt2):
    mt_data1 = mt1.get_data()
    mt_data2 = mt2.get_data()
    if mt_data1.layout != mt_data2.layout:
        raise ValueError("mt1's data and mt2's data do not have the same layout. "
                         f"mt1.get_data().layout = {mt_data1.layout} while mt2.get_data().layout = {mt_data2.layout}")

    mask = mt1.get_mask()
    mask2 = mt2.get_mask()
    if not _masks_match(mt1, mt2):
        raise ValueError("mt1 and mt2 must have matching masks")
    if mask.layout != mt2.get_mask().layout:
        raise ValueError("mt1's mask and mt2's mask do not have the same layout. "
                         f"mt1.get_mask().layout = {mask.layout} while mt2.get_mask().layout = {mask2.layout}")
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()

    if mt_data1.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_data1 = mt_data1.to_dense()
        mt_data2 = mt_data2.to_dense()
    a = mt_data1.detach().masked_fill_(~mask, 0)
    b = mt_data2.detach().masked_fill_(~mask, 0)

    if not _tensors_match(a, b, exact=False):
        raise ValueError("The data in MaskedTensor mt1 and MaskedTensor mt2 do not match")

def _create_random_mask(shape, device):
    return make_tensor(
        shape, device=device, dtype=torch.bool, low=0, high=1, requires_grad=False
    )

def _generate_sample_data(
    device="cpu", dtype=torch.float, requires_grad=True, layout=torch.strided
):
    assert layout in {
        torch.strided,
        torch.sparse_coo,
        torch.sparse_csr,
    }, "Layout must be strided/sparse_coo/sparse_csr"
    shapes = [
        [],
        [2],
        [3, 5],
        [3, 2, 1, 2],
    ]
    inputs = []
    for s in shapes:
        data = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)  # type: ignore[arg-type]
        mask = _create_random_mask(s, device)
        if layout == torch.sparse_coo:
            mask = mask.to_sparse_coo().coalesce()
            data = data.sparse_mask(mask).requires_grad_(requires_grad)
        elif layout == torch.sparse_csr:
            if data.ndim != 2 and mask.ndim != 2:
                continue
            mask = mask.to_sparse_csr()
            data = data.sparse_mask(mask)
        inputs.append(SampleInput(data, kwargs={"mask": mask}))
    return inputs


class TestBasics(TestCase):
    def sample_test(self):
        return


instantiate_parametrized_tests(TestBasics)


if __name__ == '__main__':
    run_tests()

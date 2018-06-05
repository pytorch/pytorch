# This code is MODIFIED from the version in matchbox (https://github.com/salesforce/matchbox)
# found at https://github.com/salesforce/matchbox/blob/master/matchbox/__init__.py
#
# Copyright (c) 2018, salesforce.com, inc.
#
# Licensed under the BSD 3-Clause license.
# For full license text, see https://opensource.org/licenses/BSD-3-Clause

import torch


class MaskedBatch(object):

    def __init__(self, data, mask, dims):
        if data.dim() != mask.dim() or mask.dim() != len(dims) + 1:
            raise ValueError("malformed MaskedBatch {} with:\n data: "
                             " {}\n mask: {}".format(repr(dims), repr(data), repr(mask)))
        self.data = data
        self.mask = mask
        self.dims = dims

    @classmethod
    def fromlist(cls, examples, dims):
        # TODO do some validation
        bs = len(examples)
        sizes = [max(x.size(d + 1) for x in examples)
                 for d in range(len(dims))]
        data = examples[0].new(bs, *sizes).zero_()
        mask_sizes = [s if dims[d] else 1 for d, s in enumerate(sizes)]
        mask = examples[0].new(bs, *mask_sizes).zero_().byte()
        mask.requires_grad = False
        for i, x in enumerate(examples):
            inds = [slice(0, x.size(d + 1)) if b else slice(None)
                    for d, b in enumerate(dims)]
            data[(slice(i, i + 1), *inds)] = x
            mask[(slice(i, i + 1), *inds)] = 1
        dims = torch.tensor(dims).byte()
        return cls(data, mask, dims)

    def examples(self):
        data, mask, dims = self.data, self.mask.data.long(), self.dims
        for i in range(data.size(0)):
            inds = tuple(slice(0, mask[i].sum(d, keepdim=True)[
                tuple(0 for _ in dims)])
                if b else slice(None) for d, b in enumerate(dims))
            yield data[(slice(i, i + 1), *inds)]

    def __repr__(self):
        return "MaskedBatch {} with:\n data: {}\n mask: {}".format(
            repr(self.dims), repr(self.data), repr(self.mask))

    def size(self, dim=None):
        if dim is None:
            if any(self.dims):
                raise ValueError("use size_as_tensor for dynamic dimensions")
            return self.data.size()
        if dim < 0:
            dim += self.dim()
        if dim == 0 or not self.dims[dim - 1]:
            return self.data.size(dim)
        raise ValueError("use size_as_tensor for dynamic dimensions")

    @staticmethod
    def unbind(batch, dim):
        if isinstance(batch, tuple) and len(batch) == 3:
            batch = MaskedBatch(*batch)
        if dim == 0:
            raise ValueError("cannot unbind over batch dimension")
        dims = tuple(b for d, b in enumerate(batch.dims) if d != dim - 1)
        dims = torch.tensor(dims).byte()
        if batch.dims[dim - 1]:
            return tuple((data, mask, dims)
                         for data, mask in zip(torch.unbind(batch.data, dim),
                                               torch.unbind(batch.mask, dim)))
        else:
            mask = batch.mask.squeeze(dim)
            return tuple((data, mask, dims)
                         for data in torch.unbind(batch.data, dim))

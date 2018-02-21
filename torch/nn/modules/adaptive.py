import torch

from . import Sequential, ModuleList, Linear
from .module import Module
from ..functional import log_softmax, cross_entropy


class AdaptiveLogSoftmax(Module):
    r"""Efficient softmax approximation as described in
    `Efficient softmax approximation for GPUs`_ by Edouard Grave, Armand Joulin,
     Moustapha Cissé, David Grangier, and Hervé Jégou.

    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containig less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are used.

    The idea is that the cluster that the clusters that are accessed often
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned targets.

    We highly recommend taking a look at the original paper for more details.

    ``cutoffs`` should be a Sequence of integers. It controls number of clusters
    and the partitioning of targets into clusters. For example setting
    ``cutoffs = [10, 100, 1000]`` means that first `10` targets will be assigned
    to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
    assigned to the first cluster, and targets `101, 102, ..., 1000` will be
    assigned to the second cluster, while targets
    `1001, 1002, ..., n_classes - 1` will be assigned to the last, third cluster

    ``div_value`` is used to compute the size of each additional cluster,
    which is given as :math:`\lfloor \frac{in\_features}{div\_value^i} \rfloor`,
    where :math:`i` is the cluster index (with clusters for less frequent words
    having larger indices, and indices starting at :math:`1`).

    .. warning::
        Targets passed as inputs to this module should be sorted accoridng to
        their frequency. This means that the most frequent target should be
        represented by the index `0`, and the least frequent
        target should be represented by the index `n_classes - 1`.

    .. note::
        To compute log-probabilities for all classes, the `predict_log_proba`
        method can be used.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset.
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets.
        div_value (float, optional): value used as an exponent to compute sizes
        of the clusters. Default: 2.0

    Returns:
        A Variable of size ``N``, containing computed target log probabilities
        for each example

    Shape:
        - Input: :math:`(N, in\_features)`
        - Target: :math:`(N)` where each value is `0 <= targets[i] <= C - 1`
        - Output: :math:`(N)`

    .. _Efficient softmax approximation for GPUs:
        https://arxiv.org/abs/1609.04309

    .. _Zipf's law:
        https://en.wikipedia.org/wiki/Zipf%27s_law
    """

    def __init__(self, in_features, n_classes, cutoffs, div_value=2.,
                 return_logprob=False):
        super(AdaptiveLogSoftmax, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)):
            raise ValueError('Cutoffs should be a list of unique, positive in')

        if (cutoffs != sorted(cutoffs)) \
                or (max(cutoffs) >= (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)):

            raise ValueError("Cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.return_logprob = return_logprob
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(self.in_features, self.head_size)
        self.tail = ModuleList()

        for i in range(self.n_clusters):

            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False),
                Linear(hsz, osz)
            )

            self.tail.append(projection)

    def reset_parameters(self):
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()

    def forward(self, input, target):
        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        used_rows = 0
        batch_size = target.size(0)
        out_size = batch_size if (self.return_logprob) else 1

        output = input.new(out_size).zero_()
        gather_inds = target.new(batch_size).zero_()

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            relative_target = target[target_mask] - cutoff_values[i]

            if i == 0:
                gather_inds.index_copy_(0, row_indices, relative_target)

            else:
                input_subset = input.index_select(0, row_indices)
                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                if self.return_logprob:
                    cluster_logprob = log_softmax(cluster_output, dim=1)
                    local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                    output.index_copy_(0, row_indices, local_logprob.squeeze(1))

                else:
                    output += cross_entropy(cluster_output, relative_target, size_average=False)

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     int(target.min()),
                                                     int(target.max())))

        head_output = self.head(input)

        if self.return_logprob:
            head_logprob = log_softmax(head_output, dim=1)
            output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()

        else:
            output += cross_entropy(head_output, gather_inds, size_average=False)
            output /= batch_size

        return output

    def get_log_proba(self, input):
        with torch.no_grad():
            out = input.new(input.size(0), self.n_classes).zero_()

            head_output = self.head(input)
            head_logprob = log_softmax(head_output, dim=1)

            out[:, :self.shortlist_size] += head_logprob[:, :self.shortlist_size]

            for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
                cluster_output = self.tail[i](input)
                cluster_logprob = log_softmax(cluster_output, dim=1)
                output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)

                out[:, start_idx:stop_idx] += output_logprob

        return out

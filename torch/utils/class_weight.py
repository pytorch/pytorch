import torch

def compute_sample_weight(y, class_weight=None, eps=1e-8):
    """Estimate sample weights by class for unbalanced datasets.
    Similar to sklearn.utils.class_weight.compute_sample_weight.
    Default behavior to balance classes but class_weight can be passed
    as argument.
    The weights will be normalized so their sum is equals to the number of
    samples. This normalization isn't present in sklearn's version.
    Args:
        y: 1D tensor
            Array of original class labels per sample.
        class_weight: 1D FloatTensor
            Class weights where its idx correspond to class value
        eps: float
            Value for numerical stability
    Returns:
        weights: 1D FloatTensor
           Sample weights as applied to the original y.
    Examples:
    >>> y = torch.FloatTensor([1, 0, 0, 0])
    >>> compute_sample_weight(y)
        tensor([ 2.0000,  0.6667,  0.6667,  0.6667])
    >>> compute_sample_weight(y, class_weight=torch.FloatTensor([.4, .6]))
        tensor([ 1.3333,  0.8889,  0.8889,  0.8889])
    """

    y = y.long()
    batch_size = y.size(0)

    if class_weight is None:
        n_classes = y.unique().size(0)
    else:
        n_classes = class_weight.size(0)

    y_onehot = torch.zeros(batch_size, n_classes, dtype=torch.float, device=y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    if class_weight is None:
        class_weight = 1 / (y_onehot.sum(dim=0) + eps)
        """
        classes available in y will have weight = 1/n_members
        while classes not present in y will have weight = 1/eps
        force weight of non available classes to 0
        """
        class_weight[class_weight > 1] = 0

    weights = torch.mm(y_onehot, class_weight.view(-1, 1)).squeeze()
    weights = batch_size * weights / (torch.sum(weights) + eps)
    return weights


def compute_class_weight(y, class_weight=None, eps=1e-8):
    """Estimate class weights for unbalanced datasets.
    Similar to sklearn.utils.class_weight.compute_class_weight.
    Default behavior to balance classes but class_weight can be passed
    as argument.
    The weights will be normalized so their sum is one.
    Args:
        y: 1D tensor
            Array of original class labels per sample.
        class_weight: 1D FloatTensor
            Class weights where its idx correspond to class value
        eps: float
            Value for numerical stability
    Returns:
        class_weights: 1D FloatTensor
           vector with weight for each class.
    Examples:
    >>> y = torch.FloatTensor([1, 0, 0, 0, 1, 2, 1, 1])
    >>> compute_class_weight(y)
        tensor([ 0.3333,  0.2500,  1.0000])
    >>> compute_class_weight(y, class_weight=torch.FloatTensor([.4, .5, .1]))
        tensor([ 0.4000,  0.5000,  0.1000])

    Computing weight in forward pass:
    >>> for i, (X, y) in enumerate(dataloader):
    >>>     y = y.float()
    >>>     optimizer.zero_grad()
    >>>     y_pred = model(X).squeeze()
    >>>     weights = compute_class_weight(y)
    >>>     loss = binary_cross_entropy(y_pred, y, weights)
    >>>     loss.backward()
    >>>     optimizer.step()
    """

    y = y.long()
    batch_size = y.size(0)

    if class_weight is None:
        n_classes = y.unique().size(0)
    else:
        n_classes = class_weight.size(0)

    y_onehot = torch.zeros(batch_size, n_classes, dtype=torch.float, device=y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    if class_weight is None:
        class_weight = 1 / (y_onehot.sum(dim=0) + eps)
        """
        classes available in y will have weight = 1/n_members
        while classes not present in y will have weight = 1/eps
        force weight of non available classes to 0
        """
        class_weight[class_weight > 1] = 0

    # normalize weigths
    class_weight /= class_weight.sum()
    return class_weight

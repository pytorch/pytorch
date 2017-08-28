import torch

def compute_sample_weight(y, n_classes=2, class_weight=None, eps=1e-8):
    """Estimate sample weights by class for unbalanced datasets.
    Similar to sklearn.utils.class_weight.compute_sample_weight.

    Default behavior to balance classes but class_weight can be passed
    as argument.

    The weights will be normalized so their sum is equals to the number of
    samples. This normalization isn't present in sklearn's version.

    Args:
        y: 1D tensor
            Array of original class labels per sample.
        n_classes: int
            Number of unique classes
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
        2.0000
        0.6667
        0.6667
        0.6667
        [torch.FloatTensor of size 4]

    >>> compute_sample_weight(y, n_classes=2,
                              class_weight=torch.FloatTensor([.4, .6]))
        1.3333
        0.8889
        0.8889
        0.8889
        [torch.FloatTensor of size 5]

    Computing weight in forward pass:
    >>> for i, (X, y) in enumerate(dataloader):
    >>>     y = y.float()
    >>>     X = Variable(X)
    >>>     y = Variable(y)
    >>>     optimizer.zero_grad()
    >>>     y_pred = model(X).squeeze()
    >>>     weights = compute_sample_weight(y.data) # get y tensor to use as input
    >>>     weights = Variable(weights)             # convert back to Variable to compute loss
    >>>     loss = binary_cross_entropy(y_pred, y, weights)
    >>>     loss.backward()
    >>>     optimizer.step()
    """

    y = y.long()
    batch_size = y.size(0)

    y_onehot = torch.zeros(batch_size, n_classes).float()
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

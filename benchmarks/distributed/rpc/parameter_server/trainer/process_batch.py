def process_dummy_batch(rank, batch, before):
    r"""
    A function that moves the data from CPU to GPU
    for DummyData class.
    Args:
        rank (int): worker rank
        batch (list): training sample
    """
    if before:
        batch[0] = batch[0].cuda(rank)
        batch[1] = batch[1].cuda(rank)
    else:
        batch[0] = batch[0].cpu()
        batch[1] = batch[1].cpu()
    return batch

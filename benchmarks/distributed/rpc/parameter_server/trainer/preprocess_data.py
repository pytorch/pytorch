def preprocess_dummy_data(rank, data):
    r"""
    A function that moves the data from CPU to GPU
    for DummyData class.
    Args:
        rank (int): worker rank
        data (list): training examples
    """
    for i in range(len(data)):
        data[i][0] = data[i][0].cuda(rank)
        data[i][1] = data[i][1].cuda(rank)
    return data

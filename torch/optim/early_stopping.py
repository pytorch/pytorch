class EarlyStoppingCriterion(object):
    """
    Arguments:
        patience (int): The maximum number of epochs with no improvement before early stopping should take place
        mode (str, can only be 'max' or 'min'): To take the maximum or minimum of the score for optimization
        min_delta (float, optional): Minimum change in the score to qualify as an improvement (default: 0.0)
    """

    def __init__(self, patience, mode, min_delta=0.0):
        assert patience >= 0
        assert mode in {'min', 'max'}
        assert min_delta >= 0.0
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._count = 0
        self._best_score = None
        self.is_improved = None

    def step(self, cur_score):
        """
        Checks if training should be continued given the current score.

        Arguments:
            cur_score (float): the current score

        Output:
            bool: if training should be continued
        """
        if self._best_score is None:
            self._best_score = cur_score
            return True
        else:
            if self.mode == 'max':
                self.is_improved = (cur_score >= self._best_score + self.min_delta)
            else:
                self.is_improved = (cur_score <= self._best_score - self.min_delta)

            if self.is_improved:
                self._count = 0
                self._best_score = cur_score
            else:
                self._count += 1
            return self._count <= self.patience

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

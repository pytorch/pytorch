from .monitor import Monitor


class AccuracyMonitor(Monitor):
    stat_name = 'accuracy'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', '%')
        kwargs.setdefault('precision', 2)
        super(AccuracyMonitor, self).__init__(*args, **kwargs)

    def _get_value(self, iteration, input, target, output, loss):
        batch_size = input.size(0)
        predictions = output.max(1)[1].type_as(target)
        correct = predictions.eq(target)
        if not hasattr(correct, 'sum'):
            correct = correct.cpu()
        correct = correct.sum()
        return 100. * correct / batch_size

import heapq


class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.iterations = 0
        self.stats = {}
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for i in range(1, epochs + 1):
            self.train()
            self.call_plugins('epoch', i)

    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input, batch_target = data
            self.call_plugins('batch', i, batch_input, batch_target)
            input_var = batch_input
            target_var = batch_target

            plugin_data = [None, None]

            def closure():
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', i, batch_input, batch_target,
                              *plugin_data)
            self.call_plugins('update', i, self.model)

        self.iterations += i

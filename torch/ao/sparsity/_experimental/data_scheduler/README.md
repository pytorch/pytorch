# Data Scheduler
## Intro
The data scheduler is used to control the update of the data sparsification parameters and works specifically with the data sparsifier class.
This class controls a specific config param (specified by the ```schedule_param``` argument) of
the data sparsifier class and varies it across the training process (or across time).

## API details
```BaseDataScheduler```: base class with abstract method ```get_schedule_param``` that computes the data sparsification parameter for all the data. The constructor accepts
1. ```data_sparsifier```: The data sparsifier object whose parameter will be scheduled.
2. ```schedule_param``` : a specific config of the passed data sparsifier that needs to be scheduled/varied.

```get_last_param```: gets the last scheduled parameter. Basically, a dictionary of name (of data) to schedule_param value mapping.

```step```: Applies the ```get_schedule_param``` logic every epoch/step depending on when it is called. This should always be called after the ```sparsifier.step()``` has been called.

## Write your own data scheduler
The custom data scheduler must be inherit from the ```BaseDataScheduler``` class and should have the ```get_schedule_param()``` function implemented. For example, the following scheduler halves the ```sparsity_level``` at each step.
```
class ImplementedDataScheduler(BaseDataScheduler):
    def __init__(self, data_sparsifier):
        super().__init__(data_sparsifier=data_sparsifier, schedule_param='sparsity_level')  # 'sparsity_level' is the schedule_param in this case

    def get_schedule_param(self):
        # half the sparsity level every epoch
        if self.last_epoch > 0:
            return {name: config['sparsity_level'] * 0.5
                    for name, config in self.data_sparsifier.data_groups.items()}
        else:
            return self.base_param
```

## How to use during training?

```
model = SomeModel()
optimizer = SomeOptimizer(model.parameters(), lr=...)
data_sparsifier = SomeDataSparsifier(...)

# attach data sparsifier to the model
data_sparsifier.add_data(name=..., data=..., **some_config)

data_scheduler = SomeDataScheduler(data_sparsifier, ...)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ..)

for epoch in range(EPOCHS):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        data_sparsifier.step()
        # add data_scheduler.step() here if you need schedule at each step instead of epoch

    data_scheduler.step()
    lr_scheduler.step()
```
### Note:
1. ```get_schedule_param()``` should return a dictionary wherein the keys are the names of the data and the values are the corresponding values of the ```schedule_param``` for the next step.
2. It is the responsibility of the ```BaseDataScheduler``` to call the ```get_schedule_param()``` when necessary.

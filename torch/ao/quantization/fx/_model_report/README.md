ModelReport
========

## Model Report Class in Fx Workflow

 > ⚠️ *While the example below uses the Fx Workflow, the use of the ModelReport class **does not depend** on the Fx Workflow to work*.
 The requirements are detector dependent.
 Most detectors require a **traceable GraphModule**, but some (ex. `PerChannelDetector`) require just an `nn.Module`.

#### Typical Fx Workflow
- Initialize model &rarr; Prepare model &rarr; Callibrate model &rarr; Convert model &rarr; ...

#### Fx Workflow with ModelReport
- Initialize model &rarr; Prepare model &rarr; **Add detector observers** &rarr; Callibrate model &rarr; **Generate report** &rarr; **Remove detector observers** &rarr; Convert model &rarr; ...

 > ⚠️ **You can only prepare and remove observers once with a given ModelReport Instance**: Be very careful here!

## Usage

This snippet should be ready to copy, paste, and use with the exception of a few small parts denoted in `#TODO` comments

```python
# prep model
qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
model = Model() # TODO define model
example_input = torch.randn((*args)) # TODO get example data for callibration
prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_input)

# create ModelReport instance and insert observers
detector_set = set([DynamicStaticDetector()]) # TODO add all desired detectors
model_report = ModelReport(model, detector_set)
ready_for_callibrate = model_report.prepare_detailed_callibration()

# callibrate model and generate report
ready_for_callibrate(example_input) # TODO run callibration of model with relevant data
reports = model_report.generate_model_report(remove_inserted_observers=True)
for report_name in report.keys():
    text_report, report_dict = reports[report_name]
    print(text_report, report_dict)

# Optional: we get a ModelReportVisualizer instance to do any visualizations desired
mod_rep_visualizer = tracer_reporter.generate_visualizer()
mod_rep_visualizer.generate_table_visualization() # shows collected data as a table

# TODO updated qconfig based on suggestions
```

There is a tutorial in the works that will walk through a full usage of the ModelReport API.
This tutorial will show the ModelReport API being used on toy model in both an Fx Graph Mode workflow and an alterative workflow with just a traceable model.
This README will be updated with a link to the tutorial upon completion of the tutorial.

# Key Modules Overview

## ModelReport Overview

The `ModelReport` class is the primary class the user will be interacting with in the ModelReport workflow.
There are three primary methods to be familiar with when using the ModelReport class:

- `__init__(self, model: GraphModule, desired_report_detectors: Set[DetectorBase])` constructor that takes in instances of the model we wish to generate report for (must be traceable GraphModule) and desired detectors and stores them.
This is so that we can keep track of where we want to insert observers on a detector by detector basis and also keep track of which detectors to generate reports for.
- `prepare_detailed_calibration(self)` &rarr; `GraphModule` inserts observers into the locations specified by each detector in the model.
It then returns the GraphModule with the detectors inserted into both the regular module structure as well as the node structure.
- `generate_model_report(self, remove_inserted_observers: bool)` &rarr; `Dict[str, Tuple[str, Dict]]` uses callibrated GraphModule to optionally removes inserted observers, and generate, for each detector the ModelReport instance was initialized with:
  - A string-based report that is easily digestable and actionable explaining the data collected by relevant observers for that detector
  - A dictionary containing statistics collected by the relevant observers and values calculated by the detector for further analysis or plotting

## ModelReportVisualizer Overview

After you have generated reports using the `ModelReport` instance,
you can visualize some of the collected statistics using the `ModelReportVisualizer`.
To get a `ModelReportVisualizer` instance from the `ModelReport` instance,
call `model_report.generate_visualizer()`.

When you first create the `ModelReportVisualizer` instance,
it reorganizes the reports so instead of being in a:

```
report_name
|
-- module_fqn
   |
   -- feature_name
      |
      -- feature value
```

format, it will instead be in a:
```
-- module_fqn [ordered]
   |
   -- feature_name
      |
      -- feature value
```

Essentially, all the information for each of the modules are consolidated across the different reports.
Moreover, the modules are kept in the same chronological order they would appear in the model's `forward()` method.

Then, when it comes to the visualizer, there are two main things you can do:
1. Call `mod_rep_visualizer.generate_filtered_tables()` to get a table of values you can manipulate
2. Call one of the generate visualization methods, which don't return anything but generate an output
  - `mod_rep_visualizer.generate_table_visualization()` prints out a neatly formatted table
  - `mod_rep_visualizer.generate_plot_visualization()` and `mod_rep_visualizer.generate_histogram_visualization()`
  output plots.

For both of the two things listed above, you can filter the data by either `module_fqn` or by `feature_name`.
To get a list of all the modules or features, you can call `mod_rep_visualizer.get_all_unique_module_fqns()`
and `mod_rep_visualizer.get_all_unique_feature_names()` respectively.
For the features, because some features are not plottable, you can set the flag to only get plottable features
in the aformentioned `get_all_unique_feature_names` method.

## Detector Overview

The main way to add functionality to the ModelReport API is to add more Detectors.
Detectors each have a specific focus in terms of the type of information they collect.
For example, the `DynamicStaticDetector` figures out whether Dynamic or Static Quantization is appropriate for different layers.
Meanwhile, the `InputWeightEqualizationDetector` determines whether Input-Weight Equalization should be applied for each layer.


### Requirements to Implement A Detector
All Detectors inherit from the `DetectorBase` class, and all of them (including any custom detectors you create) will need to implement 3 methods:
- `determine_observer_insert_points(self, model)` -> `Dict`: determines which observers you want to insert into a model to gather statistics and where in the model.
All of them return a dictionary mapping unique observer fully qualified names (fqns), which is where we want to insert them, to a dictionary of location and argument information in the format:

```python
return_dict = {
    "[unique_observer_fqn_of_insert_location]" :
    {
        "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node),
        "insert_observer" -> the initialized observer we wish to insert (ObserverBase),
        "insert_post" -> True if this is meant to be a post-observer for target_node, False if pre-observer,
        "observer_args" -> The arguments that are meant to be passed into the observer,
    }
}
```
- `get_detector_name(self)` -> `str`: returns the name of the detector.
You should give your detector a unique name different from existing detectors.
- `generate_detector_report(self, model)` -> `Tuple[str, Dict[str, Any]]`: generates a report based on the information the detector is trying to collect.
This report consists of both a text-based report as well as a dictionary of collected and calculated statistics.
This report is returned to the `ModelReport` instance, which will then compile all the reports of all the Detectors requested by the user.

## ModelReportObserver Overview

As seen in the [requirements to implement a detector section](#requirements-to-implement-a-detector), one of the key parts of implementing a detector is to specify what `Observer` we are trying to insert.
All the detectors in the ModelReport API use the [`ModelReportObserver`](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/_model_report/model_report_observer.py).
While the core purpose of many observers in PyTorch's Quantization API is to collect min / max information to help determine quantization parameters, the `ModelReportObserver` collects additional statistics.

The statistics collected by the `ModelReportObserver` include:
- Average batch activation range
- Epoch level activation range
- Per-channel min / max values
- Ratio of 100th percentile to some *n*th percentile
- Number of constant value batches to pass through each channel

After the `ModelReportObserver` collects the statistics above during the callibration process, the detectors then extract the information they need to generate their reports from the relevant observers.

### Using Your Own Observer

If you wish to implement your own custom Observer to use with the ModelReport API for your own custom detector, there are a few things to keep in mind.
- Make sure your detector inherits from [`torch.ao.quantization.observer.ObserverBase`](https://www.internalfb.com/code/fbsource/[20eb160510847bd24bf21a5b95092c160642155f]/fbcode/caffe2/torch/ao/quantization/observer.py?lines=122)
- In the custom detector class, come up with a descriptive and unique `PRE_OBSERVER_NAME` (and/or `POST_OBSERVER_NAME`) so that you can generate a fully qualified name (fqn) for each observer that acts a key in the returned dictionary described [here](#requirements-to-implement-a-detector)
  - [Code Example](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/_model_report/detector.py#L958)
- In the `determine_observer_insert_points()` method in your detector, initialize your custom Observer and add it to the returned dictionary described [here](#requirements-to-implement-a-detector)
  - [Code Example](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/_model_report/detector.py#L1047)

Since you are also implementing your own detector in this case, it is up to you to determine where your observers should be placed in the model, and what type of information you wish to extract from them to generate your report.

# Folder Structure

./: the main folder all the model report code is under
- `__init__.py`: File to mark ModelReport as package directory
- `detector.py`: File containing Detector classes
  - Contains `DetectorBase` class which all detectors inherit from
  - Contains several implemented detectors including:
    - `PerChannelDetector`
    - `DynamicStaticDetector`
    - `InputWeightEqualizationDetector`
    - `OutlierDetector`
- `model_report_observer.py`: File containing the `ModelReportObserver` class
  - Primary observer inserted by Detectors to collect necessary information to generate reports
- `model_report_visualizer.py`: File containing the `ModelReportVisualizer` class
  - Reorganizes reports generated by the `ModelReport` class to be:
    1. Ordered by module as they appear in a model's forward method
    2. Organized by module_fqn --> feature_name --> feature values
  - Helps generate visualizations of three different types:
    - A formatted table
    - A line plot (for both per-tensor and per-channel statistics)
    - A histogram (for both per-tensor and per-channel statistics)
- `model_report.py`: File containing the `ModelReport` class
  - Main class users are interacting with to go through the ModelReport workflow
  - API described in detail in [Overview section](#modelreport-overview)

# Tests

Tests for the ModelReport API are found in the `test_model_report_fx.py` file found [here](https://github.com/pytorch/pytorch/blob/master/test/quantization/fx/test_model_report_fx.py).

These tests include:
- Test class for the `ModelReportObserver`
- Test class for the `ModelReport` class
- Test class for the `ModelReportVisualizer` class
- Test class for **each** of the implemented Detectors

If you wish to add a Detector, make sure to create a test class modeled after one of the existing classes and test your detector.
Because users will be interacting with the Detectors through the `ModelReport` class and not directly, ensure that the tests follow this as well.

# Future Tasks and Improvements

Below is a list of tasks that can help further improve the API or bug fixes that give the API more stability:

- [ ] For DynamicStaticDetector, change method of calculating stationarity from variance to variance of variance to help account for outliers
- [ ] Add more types of visualizations for data
- [ ] Add ability to visualize histograms of histogram observers
- [ ] Automatically generate QConfigs from given suggestions
- [ ] Tune default arguments for detectors with further research and analysis on what appropriate thresholds are
- [ ] Merge the generation of the reports and the qconfig generation together
- [ ] Make a lot of the dicts returned object classes
- [ ] Change type of equalization config from `QConfigMapping` to `EqualizationMapping`

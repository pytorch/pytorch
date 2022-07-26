ModelReport
========

## Model Report Class in Fx Workflow

 > ⚠️ *While the example below uses the Fx Workflow, the use of the ModelReport class **does not depend** on the Fx Workflow to work*.
 The requirements are detector dependent.
 Most detectors require a **traceable GraphModule**, but some (ex. `PerChannelDetector`) require just a `nn.Module`.

#### Typical Fx Workflow
- Initialize model &rarr; Prepare model &rarr; Callibrate model &rarr; Convert model &rarr; ...

#### Fx Workflow with ModelReport
- Initialize model &rarr; Prepare model &rarr; **Add detector observers** &rarr; Callibrate model &rarr; **Generate report** &rarr; **Remove detector observers** &rarr; Convert model &rarr; ...

 > ⚠️ **You can only prepare and remove observers once with a given ModelReport Instance**: Be very careful here!

## Usage

This snippet should be ready to copy, paste, and use with the exception of a few small parts denoted in `#TODO` comments

```python
# prep model
q_config_mapping = torch.ao.quantization.get_default_qconfig_mapping() # alternatively use your own qconfig mapping if you alredy have one
model = Model() # TODO define model
example_input = model.get_example_data()[0] # get example data for your model
prepared_model = quantize_fx.prepare_fx(model, q_config_mapping, example_input)

# create ModelReport instance and insert observers
detector_set = set([PerChannelDetector(), InputWeightDetector(0.5), DynamicStaticDetector(), OutlierDetector()]) # TODO add all desired detectors
model_report = ModelReport(prepared_model, detector_set)
ready_for_callibrate = model_report.prepare_detailed_callibration()

# TODO run callibration of model with relavent data

# generate reports for your model and remove observers if desired
reports = model_report.generate_model_report(remove_inserted_observers=True)
for report_name in report.keys():
    text_report, report_dict = reports[report_name]
    print(text_report, report_dict)

# TODO update q_config_mapping based on feedback from reports
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
  - A string-based report that is easily digestable and actionable explaining the data collected by relavent observers for that detector
  - A dictionary containing statistics collected by the relavent observers and values calculated by the detector for futher analysis or plotting

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
        "insert_observer" -> the intialized observer we wish to insert (ObserverBase),
        "insert_post" -> True if this is meant to be a post-observer for target_node, False if pre-observer,
        "observer_args" -> The arguments that are meant to be passed into the observer,
    }
}
```
- `get_detector_name(self)` -> `str`: returns the name of the detector.
You should give your detector a unique name different from exisiting detectors.
- `generate_detector_report(self, model)` -> `Tuple[str, Dict[str, Any]]`: generates a report based on the information the detector is trying to collect.
This report consists of both a text-based report as well as a dictionary of collected and calculated statistics.
This report is returned to the `ModelReport` instance, which will then compile all the reports of all the Detectors requested by the user.

## ModelReportObserver Overview

As seen in the [requirments to implement a detector section](#requirements-to-implement-a-detector), one of the key parts of implementing a detector is to specify what `Observer` we are trying to insert.
All the detectors in the ModelReport API use the [`ModelReportObserver`](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/_model_report/model_report_observer.py).
While the core purpose of many observers in PyTorch's Quantization API is to collect min / max information to help determine quantization parameters, the `ModelReportObserver` collects additional statistics.

The statistics collected by the `ModelReportObserver` include:
- Average batch activation range
- Epoch level activation range
- Per-channel min / max values
- Ratio of 100th percentile to some *n*th percentile
- Number of constant value batches to pass through each channel

After the `ModelReportObserver` collects the statistics above during the callibration process, the detectors then extract the information they need to generate their reports from the relavent observers.

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
- `model_report.py`: File containing the `ModelReport` class
  - Main class users are interacting with to go through the ModelReport worflow
  - API described in detail in [Overview section](#modelreport-overview)

# Tests

Tests for the ModelReport API are found in the `test_model_report_fx.py` file found [here](https://github.com/pytorch/pytorch/blob/master/test/quantization/fx/test_model_report_fx.py).

These tests include:
- Test class for the `ModelReportObserver`
- Test class for the `ModelReport` class
- Test class for **each** of the implemented Detectors

If you wish to add a Detector, make sure to create a test class modeled after one of the exisiting classes and test your detector.
Because users will be interacting with the Detectors through the `ModelReport` class and not directly, ensure that the tests follow this as well.

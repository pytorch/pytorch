from typing import Dict, Set, Tuple

import torch
from torch.ao.quantization.fx._model_report.detector import DetectorBase
from torch.ao.quantization.fx.graph_module import GraphModule

class ModelReport:
    r"""
    Generates report and collects statistics
        Used to provide users suggestions on possible model configuration improvements

    Currently supports generating reports on:
    - Suggestions for dynamic vs static quantization for linear layers (Graph Modules)

    * :attr:`desired_report_detectors` The set of Detectors representing desired reports from the ModelReport class
        Make sure that these are all unique types of detectors [do not have more than 1 of the same class]

    Proper Use:
    1.) Initialize ModelReport object with reports of interest by passing in initialized detector objects
    2.) Prepare your model with prepare_fx
    3.) Call model_report.prepare_detailed_calibration on your model to add relavent observers
    4.) Callibrate your model with data
    5.) Call model_report.generate_report on your model to generate report and optionally remove added observers

    """

    def __init__(self, desired_report_detectors: Set[DetectorBase]):

        if len(desired_report_detectors) == 0:
            raise ValueError("Should include at least 1 desired report")

        # keep the reports private so they can't be modified
        self._desired_report_detectors = desired_report_detectors
        self._desired_reports = set([detector.get_detector_name() for detector in desired_report_detectors])

        # keep a mapping of desired reports to observers of interest
        # this is to get the readings, and to remove them, can create a large set
        # this set can then be used to traverse the graph and remove added observers
        self._report_name_to_observer_fqns: Dict[str, Set[str]] = {}

        # initialize each report to have empty set of observers of interest
        for desired_report in self._desired_reports:
            self._report_name_to_observer_fqns[desired_report] = set([])

    def get_desired_reports_names(self) -> Set[str]:
        """ Returns a copy of the desired reports for viewing """
        return self._desired_reports.copy()

    def get_observers_of_interest(self) -> Dict[str, Set[str]]:
        """ Returns a copy of the observers of interest for viewing """
        return self._report_name_to_observer_fqns.copy()

    def prepare_detailed_calibration(self, prepared_fx_model: GraphModule) -> GraphModule:
        r"""
        Takes in a prepared fx graph model and inserts the following observers:
        - ModelReportObserver

        Each observer is inserted based on the desired_reports into the relavent locations

        Right now, each report in self._desired_reports has independent insertions
            However, if a module already has a Observer of the same type, the insertion will not occur
            This is because all of the same type of Observer collect same information, so redundant

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns the same GraphModule with the observers inserted
        """
        pass

    def _get_node_from_fqn(self, fx_model: GraphModule, node_fqn: str) -> torch.fx.node.Node:
        r"""
        Takes in a graph model and returns the node based on the fqn

        Args
            fx_model (GraphModule): The Fx GraphModule that already contains the node with fqn node_fqn
            node_fqn (str): The fully qualified name of the node we want to find in fx_model

        Returns the Node object of the given node_fqn otherwise returns None
        """
        pass

    def generate_model_report(
        self, calibrated_fx_model: GraphModule, remove_inserted_observers: bool
    ) -> Dict[str, Tuple[str, Dict]]:
        r"""
        Takes in a callibrated fx graph model and generates all the requested reports.
        The reports generated are specified by the desired_reports specified in desired_reports

        Can optionally remove all the observers inserted by the ModelReport instance

        Args:
            calibrated_fx_model (GraphModule): The Fx GraphModule that has already been callibrated by the user
            remove_inserted_observers (bool): True to remove the observers inserted by this ModelReport instance

        Returns a mapping of each desired report name to a tuple with:
            The textual summary of that report information
            A dictionary containing relavent statistics or information for that report
        """
        pass

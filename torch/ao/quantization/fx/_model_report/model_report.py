from typing import Any, Dict, Set

import torch
from _detector import _detect_dynamic_vs_static
from torch.ao.quantization.fx.graph_module import GraphModule


DEFAULT_MODEL_REPORT_OBSERVER_PRE_EXTENSION = ".model_report_pre_observer"
DEFAULT_MODEL_REPORT_OBSERVER_POST_EXTENSION = ".model_report_post_observer"


class ModelReport:
    r"""
    Generates report and collects statistics
        Used to provide users suggestions on possible model configuration improvements

    Currently supports generating reports on:
    - Suggestions for dynamic vs static quantization for linear layers (Graph Modules)

    * :attr:`valid_reports` The set of strings representing currently supported reports for the ModelReport class
    * :attr:`desired_reports` The set of strings representing desired reports grabbed from the ModelReport class

    Proper Use:
    1.) Initialize ModelReport object with reports of interest chosen from ModelReport.valid_reports
    2.) Prepare your model with prepare_fx
    3.) Call model_report.prepare_detailed_calibration on your model to add relavent observers
    4.) Callibrate your model with data
    5.) Call model_report.generate_report on your model to generate report and optionally remove added observers

    """

    # mapping from valid reports to corresponding report generator
    valid_reports_to_detector = {
        "dynamic_vs_static": _detect_dynamic_vs_static,
    }

    # keep a accessible set of valid reports the user can request
    valid_reports: Set[str] = set(valid_reports_to_detector.keys())

    def __init__(self, desired_reports: Set[str]):
        # initialize a private mapping of possible reports to the functions to insert them
        self.__report_to_insert_method = {
            "dynamic_vs_static": self.__insert_dynamic_static_observers,
        }

        self.__report_to_supported_modules = {
            "dynamic_vs_static": set(["linear"]),
        }

        # make sure desired report strings are all valid
        for desired_report in desired_reports:
            if desired_report not in self.valid_reports:
                raise ValueError("Only select reports found in ModelReport.valid_reports")

        # keep the reports private so they can't be modified
        self.__desired_reports = desired_reports

        # keep a mapping of desired reports to observers of interest
        # this is to get the readings, and to remove them, can create a large set
        # this set can then be used to traverse the graph and remove added observers
        self.__report_to_observers_of_interest: Dict[str, Set[str]] = {}

        # initialize each report to have empty set of observers of interest
        for desired_report in self.__desired_reports:
            self.__report_to_observers_of_interest[desired_report] = set([])

    def prepare_detailed_calibration(self, prepared_fx_model: GraphModule):
        r"""
        Takes in a prepared fx graph model and inserts the following observers:
        - ModelReportObserver

        Each observer is inserted based on the desired_reports into the relavent locations

        Right now, each report in self.__desired_reports has independent insertions
            However, if a module already has a Observer of the same type, the insertion will not occur
            This is because all of the same type of Observer collect same information, so redundant

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns the same GraphModule with the observers inserted
        """
        pass

    def __insert_dynamic_static_observers(
        self, prepared_fx_model: GraphModule, modules_to_observe: Set[str]
    ):
        r"""
        Helper function for prepare_detailed_calibration

        Inserts observers for dynamic versus static quantization report and returns the model with inserted observers

        Currently inserts observers for:
            linear layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            modules_to_observe (Set[str]): The set of module types to insert observers around

        Returns the set of fqns of the observers of interest either added to or exisiting in the prepared_fx_model
        """
        pass

    def __insert_observer_around_module(
        self,
        prepared_fx_model: GraphModule,
        node_fqn: str,
        target_node: torch.fx.node.Node,
        obs_to_insert: Any,
    ):
        r"""
        Helper function that inserts the observer into both the graph structure and the module of the model

        Args
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            node_fqn (str): The fully qualified name of the node we want to insert observers around
            target_node (torch.fx.node.Node): The node in prepared_fx_module we are inserting observers around
            obs_to_insert (ObserverBase): The observer we are inserting around target_node

        Returns a set of fqns of the observers inserted by this function
        """
        pass

    def __get_node_from_fqn(self, fx_model: GraphModule, node_fqn: str) -> Any:
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
    ):
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

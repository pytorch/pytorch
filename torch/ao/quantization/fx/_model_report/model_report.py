from typing import Any, Dict, Set, Tuple

import torch
from _detector import _detect_dynamic_vs_static
from model_report_observer import ModelReportObserver
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
    # TODO should this be private
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

    def prepare_detailed_calibration(self, prepared_fx_model: GraphModule) -> GraphModule:
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

        # TODO make modules of interest something they enter or something that they specify
        # or will it be specific function to function

        # depending on the desired reports we insert observers in different places
        for report in self.__desired_reports:
            # call the relavent insertion method
            relavent_insertion_method = self.__report_to_insert_method[report]
            relavent_insertion_modules = self.__report_to_supported_modules[report]
            inserted_observers_fqns = relavent_insertion_method(prepared_fx_model, relavent_insertion_modules)

            # store inserted observers in mapping from reports to observers of interest
            self.__report_to_observers_of_interest[report].update(inserted_observers_fqns)

        # return the modified graph model
        return prepared_fx_model

    def __insert_dynamic_static_observers(
        self, prepared_fx_model: GraphModule, modules_to_observe: Set[str]
    ) -> Set[str]:
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

        # added observers of interest
        obs_of_interest = set([])

        # pick the constructor of observer to insert
        obs_ctr = ModelReportObserver

        # find all instances of the modules to observe and store them
        modules_of_interst_nodes = []

        # get the set of all node fqns currently in the Graph Module
        node_fqns = [node.target for node in prepared_fx_model.graph.nodes]

        for node in prepared_fx_model.graph.nodes:
            # see if this node is any of the modules to observe
            should_be_observed = sum(list(map(lambda x: x in node.target, modules_to_observe))) > 0

            if should_be_observed:
                # if we already have the ModelReportObserver around module, redundant in inserting another set
                pre_obs_fqn = node.target + DEFAULT_MODEL_REPORT_OBSERVER_PRE_EXTENSION
                post_obs_fqn = node.target + DEFAULT_MODEL_REPORT_OBSERVER_POST_EXTENSION

                if pre_obs_fqn not in node_fqns and post_obs_fqn not in node_fqns:
                    # add the node to our list of nodes of interest and get the fqn
                    modules_of_interst_nodes.append((node.target, node))
                else:
                    # add the observers to the set of observers relavent to this report
                    # TODO is it possible for only 1 observer to be in there, cuz we make sure to insert both?
                    obs_of_interest.add(pre_obs_fqn)
                    obs_of_interest.add(post_obs_fqn)

        # for each of the nodes, insert the observers around it
        for node_fqn, node_of_interest in modules_of_interst_nodes:
            # call helper to insert the observers around each node
            inserted_observer_fqns = self.__insert_observer_around_module(
                prepared_fx_model, node_fqn, node_of_interest, obs_ctr
            )

            # add it to the set of observers
            obs_of_interest.update(inserted_observer_fqns)

        # return all the fqns of modules of interest
        return obs_of_interest

    def __insert_observer_around_module(
        self,
        prepared_fx_model: GraphModule,
        node_fqn: str,
        target_node: torch.fx.node.Node,
        obs_to_insert: Any,
    ) -> Set[str]:
        r"""
        Helper function that inserts the observer into both the graph structure and the module of the model

        Args
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            node_fqn (str): The fully qualified name of the node we want to insert observers around
            target_node (torch.fx.node.Node): The node in prepared_fx_module we are inserting observers around
            obs_to_insert (ObserverBase): The observer we are inserting around target_node

        Returns a set of fqns of the observers inserted by this function
        """

        # keep track of observers inserted by this function
        inserted_observers = set([])

        # setup to insert the pre-observer before the target module
        with prepared_fx_model.graph.inserting_before(target_node):
            obs_to_insert = obs_to_insert()
            pre_obs_fqn = node_fqn + DEFAULT_MODEL_REPORT_OBSERVER_PRE_EXTENSION
            prepared_fx_model.add_submodule(pre_obs_fqn, obs_to_insert)
            prepared_fx_model.graph.create_node(op="call_module", target=pre_obs_fqn, args=target_node.args)
            inserted_observers.add(pre_obs_fqn)

        # setup to insert the post-observer after the target module
        with prepared_fx_model.graph.inserting_after(target_node):
            obs_to_insert = obs_to_insert()
            post_obs_fqn = node_fqn + DEFAULT_MODEL_REPORT_OBSERVER_POST_EXTENSION
            prepared_fx_model.add_submodule(post_obs_fqn, obs_to_insert)
            prepared_fx_model.graph.create_node(op="call_module", target=post_obs_fqn, args=(target_node,))
            inserted_observers.add(post_obs_fqn)

        # recompile model after inserts are made
        prepared_fx_model.recompile()

        return inserted_observers

    def __get_node_from_fqn(self, fx_model: GraphModule, node_fqn: str) -> Any:
        r"""
        Takes in a graph model and returns the node based on the fqn

        Args
            fx_model (GraphModule): The Fx GraphModule that already contains the node with fqn node_fqn
            node_fqn (str): The fully qualified name of the node we want to find in fx_model

        Returns the Node object of the given node_fqn otherwise returns None
        """

        node_to_return = None
        for node in fx_model.graph.nodes:
            # if the target matches the fqn, it's the node we are looking for
            if node.target == node_fqn:
                node_to_return = node
                break

        return node_to_return

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
        # keep track of all the reports of interest and their outputs
        reports_of_interest = {}

        for desired_report in self.__desired_reports:
            # generate the report and collect the outputs
            report_generator_function = self.valid_reports_to_detector[desired_report]
            report_output = report_generator_function(calibrated_fx_model)
            reports_of_interest[desired_report] = report_output

        if remove_inserted_observers:
            # we go through and remove all the ModelReport observers inserted
            # TODO should it be modified so it only removes those that were inserted by this instance of the class?

            # get the set of all Observers inserted by this instance of ModelReport
            all_observers_of_interest: Set[str] = set([])
            for desired_report, observers_of_interest in self.__report_to_observers_of_interest.values():
                all_observers_of_interest.update(observers_of_interest)

            # go through all_observers_of_interest and remove them from the graph and model
            for observer_fqn in all_observers_of_interest:
                # remove the observer from the model
                calibrated_fx_model.delete_submodule(observer_fqn)

                # remove the observer from the graph structure
                node_obj = self.__get_node_from_fqn(calibrated_fx_model, observer_fqn)

            # remember to recompile the model
            calibrated_fx_model.recompile()

        # return the reports of interest
        return reports_of_interest

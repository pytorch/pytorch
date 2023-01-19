import collections

from .compat import collections_abc
from .providers import AbstractResolver
from .structs import DirectedGraph


RequirementInformation = collections.namedtuple(
    "RequirementInformation", ["requirement", "parent"]
)


class ResolverException(Exception):
    """A base class for all exceptions raised by this module.

    Exceptions derived by this class should all be handled in this module. Any
    bubbling pass the resolver should be treated as a bug.
    """


class RequirementsConflicted(ResolverException):
    def __init__(self, criterion):
        super(RequirementsConflicted, self).__init__(criterion)
        self.criterion = criterion

    def __str__(self):
        return "Requirements conflict: {}".format(
            ", ".join(repr(r) for r in self.criterion.iter_requirement()),
        )


class InconsistentCandidate(ResolverException):
    def __init__(self, candidate, criterion):
        super(InconsistentCandidate, self).__init__(candidate, criterion)
        self.candidate = candidate
        self.criterion = criterion

    def __str__(self):
        return "Provided candidate {!r} does not satisfy {}".format(
            self.candidate,
            ", ".join(repr(r) for r in self.criterion.iter_requirement()),
        )


class Criterion(object):
    """Representation of possible resolution results of a package.

    This holds three attributes:

    * `information` is a collection of `RequirementInformation` pairs.
      Each pair is a requirement contributing to this criterion, and the
      candidate that provides the requirement.
    * `incompatibilities` is a collection of all known not-to-work candidates
      to exclude from consideration.
    * `candidates` is a collection containing all possible candidates deducted
      from the union of contributing requirements and known incompatibilities.
      It should never be empty, except when the criterion is an attribute of a
      raised `RequirementsConflicted` (in which case it is always empty).

    .. note::
        This class is intended to be externally immutable. **Do not** mutate
        any of its attribute containers.
    """

    def __init__(self, candidates, information, incompatibilities):
        self.candidates = candidates
        self.information = information
        self.incompatibilities = incompatibilities

    def __repr__(self):
        requirements = ", ".join(
            "({!r}, via={!r})".format(req, parent)
            for req, parent in self.information
        )
        return "Criterion({})".format(requirements)

    @classmethod
    def from_requirement(cls, provider, requirement, parent):
        """Build an instance from a requirement.
        """
        candidates = provider.find_matches([requirement])
        if not isinstance(candidates, collections_abc.Sequence):
            candidates = list(candidates)
        criterion = cls(
            candidates=candidates,
            information=[RequirementInformation(requirement, parent)],
            incompatibilities=[],
        )
        if not candidates:
            raise RequirementsConflicted(criterion)
        return criterion

    def iter_requirement(self):
        return (i.requirement for i in self.information)

    def iter_parent(self):
        return (i.parent for i in self.information)

    def merged_with(self, provider, requirement, parent):
        """Build a new instance from this and a new requirement.
        """
        infos = list(self.information)
        infos.append(RequirementInformation(requirement, parent))
        candidates = provider.find_matches([r for r, _ in infos])
        if not isinstance(candidates, collections_abc.Sequence):
            candidates = list(candidates)
        criterion = type(self)(candidates, infos, list(self.incompatibilities))
        if not candidates:
            raise RequirementsConflicted(criterion)
        return criterion

    def excluded_of(self, candidate):
        """Build a new instance from this, but excluding specified candidate.

        Returns the new instance, or None if we still have no valid candidates.
        """
        incompats = list(self.incompatibilities)
        incompats.append(candidate)
        candidates = [c for c in self.candidates if c != candidate]
        if not candidates:
            return None
        criterion = type(self)(candidates, list(self.information), incompats)
        return criterion


class ResolutionError(ResolverException):
    pass


class ResolutionImpossible(ResolutionError):
    def __init__(self, causes):
        super(ResolutionImpossible, self).__init__(causes)
        # causes is a list of RequirementInformation objects
        self.causes = causes


class ResolutionTooDeep(ResolutionError):
    def __init__(self, round_count):
        super(ResolutionTooDeep, self).__init__(round_count)
        self.round_count = round_count


# Resolution state in a round.
State = collections.namedtuple("State", "mapping criteria")


class Resolution(object):
    """Stateful resolution object.

    This is designed as a one-off object that holds information to kick start
    the resolution process, and holds the results afterwards.
    """

    def __init__(self, provider, reporter):
        self._p = provider
        self._r = reporter
        self._states = []

    @property
    def state(self):
        try:
            return self._states[-1]
        except IndexError:
            raise AttributeError("state")

    def _push_new_state(self):
        """Push a new state into history.

        This new state will be used to hold resolution results of the next
        coming round.
        """
        try:
            base = self._states[-1]
        except IndexError:
            state = State(mapping=collections.OrderedDict(), criteria={})
        else:
            state = State(
                mapping=base.mapping.copy(), criteria=base.criteria.copy(),
            )
        self._states.append(state)

    def _merge_into_criterion(self, requirement, parent):
        self._r.adding_requirement(requirement, parent)
        name = self._p.identify(requirement)
        try:
            crit = self.state.criteria[name]
        except KeyError:
            crit = Criterion.from_requirement(self._p, requirement, parent)
        else:
            crit = crit.merged_with(self._p, requirement, parent)
        return name, crit

    def _get_criterion_item_preference(self, item):
        name, criterion = item
        try:
            pinned = self.state.mapping[name]
        except KeyError:
            pinned = None
        return self._p.get_preference(
            pinned, criterion.candidates, criterion.information,
        )

    def _is_current_pin_satisfying(self, name, criterion):
        try:
            current_pin = self.state.mapping[name]
        except KeyError:
            return False
        return all(
            self._p.is_satisfied_by(r, current_pin)
            for r in criterion.iter_requirement()
        )

    def _get_criteria_to_update(self, candidate):
        criteria = {}
        for r in self._p.get_dependencies(candidate):
            name, crit = self._merge_into_criterion(r, parent=candidate)
            criteria[name] = crit
        return criteria

    def _attempt_to_pin_criterion(self, name, criterion):
        causes = []
        for candidate in criterion.candidates:
            try:
                criteria = self._get_criteria_to_update(candidate)
            except RequirementsConflicted as e:
                causes.append(e.criterion)
                continue

            # Check the newly-pinned candidate actually works. This should
            # always pass under normal circumstances, but in the case of a
            # faulty provider, we will raise an error to notify the implementer
            # to fix find_matches() and/or is_satisfied_by().
            satisfied = all(
                self._p.is_satisfied_by(r, candidate)
                for r in criterion.iter_requirement()
            )
            if not satisfied:
                raise InconsistentCandidate(candidate, criterion)

            # Put newly-pinned candidate at the end. This is essential because
            # backtracking looks at this mapping to get the last pin.
            self._r.pinning(candidate)
            self.state.mapping.pop(name, None)
            self.state.mapping[name] = candidate
            self.state.criteria.update(criteria)

            return []

        # All candidates tried, nothing works. This criterion is a dead
        # end, signal for backtracking.
        return causes

    def _backtrack(self):
        # Drop the current state, it's known not to work.
        del self._states[-1]

        # We need at least 2 states here:
        # (a) One to backtrack to.
        # (b) One to restore state (a) to its state prior to candidate-pinning,
        #     so we can pin another one instead.

        while len(self._states) >= 2:
            # Retract the last candidate pin.
            prev_state = self._states.pop()
            try:
                name, candidate = prev_state.mapping.popitem()
            except KeyError:
                continue
            self._r.backtracking(candidate)

            # Create a new state to work on, with the newly known not-working
            # candidate excluded.
            self._push_new_state()

            # Mark the retracted candidate as incompatible.
            criterion = self.state.criteria[name].excluded_of(candidate)
            if criterion is None:
                # This state still does not work. Try the still previous state.
                del self._states[-1]
                continue
            self.state.criteria[name] = criterion

            return True

        return False

    def resolve(self, requirements, max_rounds):
        if self._states:
            raise RuntimeError("already resolved")

        self._push_new_state()
        for r in requirements:
            try:
                name, crit = self._merge_into_criterion(r, parent=None)
            except RequirementsConflicted as e:
                raise ResolutionImpossible(e.criterion.information)
            self.state.criteria[name] = crit

        self._r.starting()

        for round_index in range(max_rounds):
            self._r.starting_round(round_index)

            self._push_new_state()
            curr = self.state

            unsatisfied_criterion_items = [
                item
                for item in self.state.criteria.items()
                if not self._is_current_pin_satisfying(*item)
            ]

            # All criteria are accounted for. Nothing more to pin, we are done!
            if not unsatisfied_criterion_items:
                del self._states[-1]
                self._r.ending(curr)
                return self.state

            # Choose the most preferred unpinned criterion to try.
            name, criterion = min(
                unsatisfied_criterion_items,
                key=self._get_criterion_item_preference,
            )
            failure_causes = self._attempt_to_pin_criterion(name, criterion)

            # Backtrack if pinning fails.
            if failure_causes:
                result = self._backtrack()
                if not result:
                    causes = [
                        i for crit in failure_causes for i in crit.information
                    ]
                    raise ResolutionImpossible(causes)

            self._r.ending_round(round_index, curr)

        raise ResolutionTooDeep(max_rounds)


def _has_route_to_root(criteria, key, all_keys, connected):
    if key in connected:
        return True
    if key not in criteria:
        return False
    for p in criteria[key].iter_parent():
        try:
            pkey = all_keys[id(p)]
        except KeyError:
            continue
        if pkey in connected:
            connected.add(key)
            return True
        if _has_route_to_root(criteria, pkey, all_keys, connected):
            connected.add(key)
            return True
    return False


Result = collections.namedtuple("Result", "mapping graph criteria")


def _build_result(state):
    mapping = state.mapping
    all_keys = {id(v): k for k, v in mapping.items()}
    all_keys[id(None)] = None

    graph = DirectedGraph()
    graph.add(None)  # Sentinel as root dependencies' parent.

    connected = {None}
    for key, criterion in state.criteria.items():
        if not _has_route_to_root(state.criteria, key, all_keys, connected):
            continue
        if key not in graph:
            graph.add(key)
        for p in criterion.iter_parent():
            try:
                pkey = all_keys[id(p)]
            except KeyError:
                continue
            if pkey not in graph:
                graph.add(pkey)
            graph.connect(pkey, key)

    return Result(
        mapping={k: v for k, v in mapping.items() if k in connected},
        graph=graph,
        criteria=state.criteria,
    )


class Resolver(AbstractResolver):
    """The thing that performs the actual resolution work.
    """

    base_exception = ResolverException

    def resolve(self, requirements, max_rounds=100):
        """Take a collection of constraints, spit out the resolution result.

        The return value is a representation to the final resolution result. It
        is a tuple subclass with three public members:

        * `mapping`: A dict of resolved candidates. Each key is an identifier
            of a requirement (as returned by the provider's `identify` method),
            and the value is the resolved candidate.
        * `graph`: A `DirectedGraph` instance representing the dependency tree.
            The vertices are keys of `mapping`, and each edge represents *why*
            a particular package is included. A special vertex `None` is
            included to represent parents of user-supplied requirements.
        * `criteria`: A dict of "criteria" that hold detailed information on
            how edges in the graph are derived. Each key is an identifier of a
            requirement, and the value is a `Criterion` instance.

        The following exceptions may be raised if a resolution cannot be found:

        * `ResolutionImpossible`: A resolution cannot be found for the given
            combination of requirements. The `causes` attribute of the
            exception is a list of (requirement, parent), giving the
            requirements that could not be satisfied.
        * `ResolutionTooDeep`: The dependency tree is too deeply nested and
            the resolver gave up. This is usually caused by a circular
            dependency, but you can try to resolve this by increasing the
            `max_rounds` argument.
        """
        resolution = Resolution(self.provider, self.reporter)
        state = resolution.resolve(requirements, max_rounds=max_rounds)
        return _build_result(state)

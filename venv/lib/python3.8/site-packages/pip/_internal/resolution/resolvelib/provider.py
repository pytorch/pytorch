from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.resolvelib.providers import AbstractProvider

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import (
        Any,
        Dict,
        Iterable,
        Optional,
        Sequence,
        Set,
        Tuple,
        Union,
    )

    from .base import Requirement, Candidate
    from .factory import Factory

# Notes on the relationship between the provider, the factory, and the
# candidate and requirement classes.
#
# The provider is a direct implementation of the resolvelib class. Its role
# is to deliver the API that resolvelib expects.
#
# Rather than work with completely abstract "requirement" and "candidate"
# concepts as resolvelib does, pip has concrete classes implementing these two
# ideas. The API of Requirement and Candidate objects are defined in the base
# classes, but essentially map fairly directly to the equivalent provider
# methods. In particular, `find_matches` and `is_satisfied_by` are
# requirement methods, and `get_dependencies` is a candidate method.
#
# The factory is the interface to pip's internal mechanisms. It is stateless,
# and is created by the resolver and held as a property of the provider. It is
# responsible for creating Requirement and Candidate objects, and provides
# services to those objects (access to pip's finder and preparer).


class PipProvider(AbstractProvider):
    def __init__(
        self,
        factory,  # type: Factory
        constraints,  # type: Dict[str, SpecifierSet]
        ignore_dependencies,  # type: bool
        upgrade_strategy,  # type: str
        user_requested,  # type: Set[str]
    ):
        # type: (...) -> None
        self._factory = factory
        self._constraints = constraints
        self._ignore_dependencies = ignore_dependencies
        self._upgrade_strategy = upgrade_strategy
        self.user_requested = user_requested

    def _sort_matches(self, matches):
        # type: (Iterable[Candidate]) -> Sequence[Candidate]

        # The requirement is responsible for returning a sequence of potential
        # candidates, one per version. The provider handles the logic of
        # deciding the order in which these candidates should be passed to
        # the resolver.

        # The `matches` argument is a sequence of candidates, one per version,
        # which are potential options to be installed. The requirement will
        # have already sorted out whether to give us an already-installed
        # candidate or a version from PyPI (i.e., it will deal with options
        # like --force-reinstall and --ignore-installed).

        # We now work out the correct order.
        #
        # 1. If no other considerations apply, later versions take priority.
        # 2. An already installed distribution is preferred over any other,
        #    unless the user has requested an upgrade.
        #    Upgrades are allowed when:
        #    * The --upgrade flag is set, and
        #      - The project was specified on the command line, or
        #      - The project is a dependency and the "eager" upgrade strategy
        #        was requested.
        def _eligible_for_upgrade(name):
            # type: (str) -> bool
            """Are upgrades allowed for this project?

            This checks the upgrade strategy, and whether the project was one
            that the user specified in the command line, in order to decide
            whether we should upgrade if there's a newer version available.

            (Note that we don't need access to the `--upgrade` flag, because
            an upgrade strategy of "to-satisfy-only" means that `--upgrade`
            was not specified).
            """
            if self._upgrade_strategy == "eager":
                return True
            elif self._upgrade_strategy == "only-if-needed":
                return (name in self.user_requested)
            return False

        def sort_key(c):
            # type: (Candidate) -> int
            """Return a sort key for the matches.

            The highest priority should be given to installed candidates that
            are not eligible for upgrade. We use the integer value in the first
            part of the key to sort these before other candidates.

            We only pull the installed candidate to the bottom (i.e. most
            preferred), but otherwise keep the ordering returned by the
            requirement. The requirement is responsible for returning a list
            otherwise sorted for the resolver, taking account for versions
            and binary preferences as specified by the user.
            """
            if c.is_installed and not _eligible_for_upgrade(c.name):
                return 1
            return 0

        return sorted(matches, key=sort_key)

    def identify(self, dependency):
        # type: (Union[Requirement, Candidate]) -> str
        return dependency.name

    def get_preference(
        self,
        resolution,  # type: Optional[Candidate]
        candidates,  # type: Sequence[Candidate]
        information  # type: Sequence[Tuple[Requirement, Candidate]]
    ):
        # type: (...) -> Any
        # Use the "usual" value for now
        return len(candidates)

    def find_matches(self, requirements):
        # type: (Sequence[Requirement]) -> Iterable[Candidate]
        if not requirements:
            return []
        constraint = self._constraints.get(
            requirements[0].name, SpecifierSet(),
        )
        candidates = self._factory.find_candidates(requirements, constraint)
        return reversed(self._sort_matches(candidates))

    def is_satisfied_by(self, requirement, candidate):
        # type: (Requirement, Candidate) -> bool
        return requirement.is_satisfied_by(candidate)

    def get_dependencies(self, candidate):
        # type: (Candidate) -> Sequence[Requirement]
        with_requires = not self._ignore_dependencies
        return [
            r
            for r in candidate.iter_dependencies(with_requires)
            if r is not None
        ]

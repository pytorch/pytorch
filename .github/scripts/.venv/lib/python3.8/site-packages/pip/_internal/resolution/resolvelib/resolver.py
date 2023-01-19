import functools
import logging

from pip._vendor import six
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.resolvelib import BaseReporter, ResolutionImpossible
from pip._vendor.resolvelib import Resolver as RLResolver

from pip._internal.exceptions import InstallationError
from pip._internal.req.req_install import check_invalid_constraint_type
from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver
from pip._internal.resolution.resolvelib.provider import PipProvider
from pip._internal.utils.misc import dist_is_editable
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

from .factory import Factory

if MYPY_CHECK_RUNNING:
    from typing import Dict, List, Optional, Set, Tuple

    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.resolvelib.resolvers import Result
    from pip._vendor.resolvelib.structs import Graph

    from pip._internal.cache import WheelCache
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.operations.prepare import RequirementPreparer
    from pip._internal.req.req_install import InstallRequirement
    from pip._internal.resolution.base import InstallRequirementProvider


logger = logging.getLogger(__name__)


class Resolver(BaseResolver):
    _allowed_strategies = {"eager", "only-if-needed", "to-satisfy-only"}

    def __init__(
        self,
        preparer,  # type: RequirementPreparer
        finder,  # type: PackageFinder
        wheel_cache,  # type: Optional[WheelCache]
        make_install_req,  # type: InstallRequirementProvider
        use_user_site,  # type: bool
        ignore_dependencies,  # type: bool
        ignore_installed,  # type: bool
        ignore_requires_python,  # type: bool
        force_reinstall,  # type: bool
        upgrade_strategy,  # type: str
        py_version_info=None,  # type: Optional[Tuple[int, ...]]
        lazy_wheel=False,  # type: bool
    ):
        super(Resolver, self).__init__()
        if lazy_wheel:
            logger.warning(
                'pip is using lazily downloaded wheels using HTTP '
                'range requests to obtain dependency information. '
                'This experimental feature is enabled through '
                '--use-feature=fast-deps and it is not ready for production.'
            )

        assert upgrade_strategy in self._allowed_strategies

        self.factory = Factory(
            finder=finder,
            preparer=preparer,
            make_install_req=make_install_req,
            wheel_cache=wheel_cache,
            use_user_site=use_user_site,
            force_reinstall=force_reinstall,
            ignore_installed=ignore_installed,
            ignore_requires_python=ignore_requires_python,
            py_version_info=py_version_info,
            lazy_wheel=lazy_wheel,
        )
        self.ignore_dependencies = ignore_dependencies
        self.upgrade_strategy = upgrade_strategy
        self._result = None  # type: Optional[Result]

    def resolve(self, root_reqs, check_supported_wheels):
        # type: (List[InstallRequirement], bool) -> RequirementSet

        constraints = {}  # type: Dict[str, SpecifierSet]
        user_requested = set()  # type: Set[str]
        requirements = []
        for req in root_reqs:
            if req.constraint:
                # Ensure we only accept valid constraints
                problem = check_invalid_constraint_type(req)
                if problem:
                    raise InstallationError(problem)

                name = canonicalize_name(req.name)
                if name in constraints:
                    constraints[name] = constraints[name] & req.specifier
                else:
                    constraints[name] = req.specifier
            else:
                if req.user_supplied and req.name:
                    user_requested.add(canonicalize_name(req.name))
                r = self.factory.make_requirement_from_install_req(
                    req, requested_extras=(),
                )
                if r is not None:
                    requirements.append(r)

        provider = PipProvider(
            factory=self.factory,
            constraints=constraints,
            ignore_dependencies=self.ignore_dependencies,
            upgrade_strategy=self.upgrade_strategy,
            user_requested=user_requested,
        )
        reporter = BaseReporter()
        resolver = RLResolver(provider, reporter)

        try:
            try_to_avoid_resolution_too_deep = 2000000
            self._result = resolver.resolve(
                requirements, max_rounds=try_to_avoid_resolution_too_deep,
            )

        except ResolutionImpossible as e:
            error = self.factory.get_installation_error(e)
            six.raise_from(error, e)

        req_set = RequirementSet(check_supported_wheels=check_supported_wheels)
        for candidate in self._result.mapping.values():
            ireq = candidate.get_install_requirement()
            if ireq is None:
                continue

            # Check if there is already an installation under the same name,
            # and set a flag for later stages to uninstall it, if needed.
            # * There isn't, good -- no uninstalltion needed.
            # * The --force-reinstall flag is set. Always reinstall.
            # * The installation is different in version or editable-ness, so
            #   we need to uninstall it to install the new distribution.
            # * The installed version is the same as the pending distribution.
            #   Skip this distrubiton altogether to save work.
            installed_dist = self.factory.get_dist_to_uninstall(candidate)
            if installed_dist is None:
                ireq.should_reinstall = False
            elif self.factory.force_reinstall:
                ireq.should_reinstall = True
            elif installed_dist.parsed_version != candidate.version:
                ireq.should_reinstall = True
            elif dist_is_editable(installed_dist) != candidate.is_editable:
                ireq.should_reinstall = True
            else:
                continue

            link = candidate.source_link
            if link and link.is_yanked:
                # The reason can contain non-ASCII characters, Unicode
                # is required for Python 2.
                msg = (
                    u'The candidate selected for download or install is a '
                    u'yanked version: {name!r} candidate (version {version} '
                    u'at {link})\nReason for being yanked: {reason}'
                ).format(
                    name=candidate.name,
                    version=candidate.version,
                    link=link,
                    reason=link.yanked_reason or u'<none given>',
                )
                logger.warning(msg)

            req_set.add_named_requirement(ireq)

        return req_set

    def get_installation_order(self, req_set):
        # type: (RequirementSet) -> List[InstallRequirement]
        """Get order for installation of requirements in RequirementSet.

        The returned list contains a requirement before another that depends on
        it. This helps ensure that the environment is kept consistent as they
        get installed one-by-one.

        The current implementation creates a topological ordering of the
        dependency graph, while breaking any cycles in the graph at arbitrary
        points. We make no guarantees about where the cycle would be broken,
        other than they would be broken.
        """
        assert self._result is not None, "must call resolve() first"

        graph = self._result.graph
        weights = get_topological_weights(graph)

        sorted_items = sorted(
            req_set.requirements.items(),
            key=functools.partial(_req_set_item_sorter, weights=weights),
            reverse=True,
        )
        return [ireq for _, ireq in sorted_items]


def get_topological_weights(graph):
    # type: (Graph) -> Dict[Optional[str], int]
    """Assign weights to each node based on how "deep" they are.

    This implementation may change at any point in the future without prior
    notice.

    We take the length for the longest path to any node from root, ignoring any
    paths that contain a single node twice (i.e. cycles). This is done through
    a depth-first search through the graph, while keeping track of the path to
    the node.

    Cycles in the graph result would result in node being revisited while also
    being it's own path. In this case, take no action. This helps ensure we
    don't get stuck in a cycle.

    When assigning weight, the longer path (i.e. larger length) is preferred.
    """
    path = set()  # type: Set[Optional[str]]
    weights = {}  # type: Dict[Optional[str], int]

    def visit(node):
        # type: (Optional[str]) -> None
        if node in path:
            # We hit a cycle, so we'll break it here.
            return

        # Time to visit the children!
        path.add(node)
        for child in graph.iter_children(node):
            visit(child)
        path.remove(node)

        last_known_parent_count = weights.get(node, 0)
        weights[node] = max(last_known_parent_count, len(path))

    # `None` is guaranteed to be the root node by resolvelib.
    visit(None)

    # Sanity checks
    assert weights[None] == 0
    assert len(weights) == len(graph)

    return weights


def _req_set_item_sorter(
    item,     # type: Tuple[str, InstallRequirement]
    weights,  # type: Dict[Optional[str], int]
):
    # type: (...) -> Tuple[int, str]
    """Key function used to sort install requirements for installation.

    Based on the "weight" mapping calculated in ``get_installation_order()``.
    The canonical package name is returned as the second member as a tie-
    breaker to ensure the result is predictable, which is useful in tests.
    """
    name = canonicalize_name(item[0])
    return weights[name], name

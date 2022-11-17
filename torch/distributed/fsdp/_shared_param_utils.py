import collections

from typing import Dict, List, NamedTuple, Set, Tuple

import torch.nn as nn


class SharedParamInfo(NamedTuple):
    """
    This includes information about shared parameters needed for high-level
    FSDP initialization (i.e. choosing which parameters to flatten together).
    This differs from the shared parameter info needed for ``FlatParamHandle``
    initialization (where the parameters to flatten have already been chosen).
    """

    module1: nn.Module
    module2: nn.Module
    param: nn.Parameter


def get_shared_param_info_to_lca(
    root_module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> Dict[SharedParamInfo, nn.Module]:
    """
    Computes the lowest common ancestor modules for the queries encoded by
    ``shared_param_infos``, where each entry in the list gives two modules,
    which each represent a vertex in the module tree.

    This follows a simple version of Tarjan's offline lowest common ancestor
    algorithm based on a union-find data structure.
    """
    shared_param_infos = get_shared_param_infos(root_module, ignored_params)
    # Construct edge list representing the LCA query graph
    module_to_sharing_modules: Dict[
        nn.Module, List[nn.Module]
    ] = collections.defaultdict(list)
    for module1, module2, _ in shared_param_infos:
        module_to_sharing_modules[module1].append(module2)
        module_to_sharing_modules[module2].append(module1)

    module_to_parent: Dict[nn.Module, nn.Module] = {}
    module_to_rank: Dict[nn.Module, int] = {}
    module_to_ancestor: Dict[nn.Module, nn.Module] = {}
    visited_modules: Set[nn.Module] = set()
    lca_query_to_lca: Dict[Tuple[nn.Module, nn.Module], nn.Module] = {}

    def tarjan_lca(module: nn.Module):
        make_set(module, module_to_parent, module_to_rank)
        for child_module in module.children():
            tarjan_lca(child_module)
            union(module, child_module, module_to_parent, module_to_rank)
            module_to_ancestor[find(module, module_to_parent)] = module
        visited_modules.add(module)
        if module not in module_to_sharing_modules:
            return
        for other_module in module_to_sharing_modules[module]:
            if other_module in visited_modules:
                lca_query = get_lca_query(module, other_module)
                if lca_query in lca_query_to_lca:
                    continue  # already computed
                lca_query_to_lca[lca_query] = module_to_ancestor[
                    find(other_module, module_to_parent)
                ]

    tarjan_lca(root_module)  # compute the LCAs
    shared_param_info_to_lca: Dict[SharedParamInfo, nn.Module] = {}
    for shared_param_info in shared_param_infos:
        module1, module2, _ = shared_param_info
        lca_query = get_lca_query(module1, module2)
        assert lca_query in lca_query_to_lca, f"Missing LCA query: {lca_query}"
        shared_param_info_to_lca[shared_param_info] = lca_query_to_lca[lca_query]
    return shared_param_info_to_lca


def get_shared_param_infos(
    root_module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> List[SharedParamInfo]:
    """
    Returns a list of shared parameter information in the module tree rooted at
    ``root_module`` and ignoring ``ignored_params``.
    """
    visited_param_to_module: Dict[nn.Parameter, nn.Module] = {}
    shared_param_infos: List[SharedParamInfo] = []
    for module in root_module.modules():
        for param in module.parameters(recurse=False):
            if param in ignored_params:
                continue
            if param in visited_param_to_module:  # shared parameter
                query = get_lca_query(visited_param_to_module[param], module)
                shared_param_infos.append(SharedParamInfo(*query, param))
            else:
                visited_param_to_module[param] = module
    return shared_param_infos


def get_lca_query(
    module1: nn.Module,
    module2: nn.Module,
) -> Tuple[nn.Module, nn.Module]:
    """
    Returns an LCA query, where we fix an order for any two modules to avoid
    duplicates due to reordering and to ensure consistent dict lookups.
    """
    return (module1, module2) if id(module1) < id(module2) else (module2, module1)


def make_set(
    module: nn.Module,
    module_to_parent: Dict[nn.Module, nn.Module],
    module_to_rank: Dict[nn.Module, int],
) -> None:
    module_to_parent[module] = module
    module_to_rank[module] = 1


def union(
    module1: nn.Module,
    module2: nn.Module,
    module_to_parent: Dict[nn.Module, nn.Module],
    module_to_rank: Dict[nn.Module, int],
) -> None:
    root1 = find(module1, module_to_parent)
    root2 = find(module2, module_to_parent)
    if module_to_rank[root1] > module_to_rank[root2]:
        module_to_parent[root2] = root1
    elif module_to_rank[root1] < module_to_rank[root2]:
        module_to_parent[root1] = root2
    else:
        module_to_parent[root2] = root1
        module_to_rank[root1] += 1


def find(
    module: nn.Module,
    module_to_parent: Dict[nn.Module, nn.Module],
) -> nn.Module:
    if module_to_parent[module] != module:
        module_to_parent[module] = find(module_to_parent[module], module_to_parent)
    return module_to_parent[module]

#!/usr/bin/env python3
"""C++ API documentation coverage checker.

Auto-discovers public C++ APIs from Doxygen XML output and checks which ones
are documented in the RST source files via Breathe or Sphinx C++ domain
directives.

Uses an exclusion list (EXCLUDED_APIS) to skip internal/detail symbols that
don't need public documentation, rather than maintaining a hardcoded allowlist.

Additionally checks built HTML for broken formatting (empty pages,
unresolved directives, rendering errors).

Usage:
    python check_coverage.py                  # RST coverage + HTML checks
    python check_coverage.py --coverxygen     # also run coverxygen on Doxygen XML
"""

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "source"
BUILD_HTML = SCRIPT_DIR / "build" / "html"
BUILD_XML = SCRIPT_DIR / "build" / "xml"
COVERAGE_OUTPUT = SCRIPT_DIR / "cpp_coverage.txt"
HTML_REPORT = SCRIPT_DIR / "cpp_html_issues.txt"

# ─── Inclusion override ──────────────────────────────────────────────────────
# Symbols that match an exclusion pattern but should still be tracked.
# Use this for "internal" APIs that are widely used as public API.
# Add the fully-qualified symbol name here and it will bypass all exclusions.
INCLUDED_SYMBOLS: set[str] = {
    # Example: "c10::IValue" would track it even though c10::IValue is excluded
}

# ─── Exclusion list ──────────────────────────────────────────────────────────
# Symbols that should NOT be flagged as missing documentation.
# Add internal, detail, or otherwise non-public symbols here.
# Note: INCLUDED_SYMBOLS takes priority over these exclusions.

EXCLUDED_PATTERNS = [
    # Internal/detail namespaces
    r".*::detail::.*",
    r".*::detail_::.*",
    r"torch::python::.*",
    # Underscore-prefixed internal classes
    r".*::_\w+",
    # Enum helper structs
    r"torch::enumtype::.*",
    # OptimizerCloneableOptions SFINAE helpers
    r"torch::optim::OptimizerCloneableOptions::.*",
    # Internal optimizer state/options cloneable helpers
    r"torch::optim::OptimizerCloneable.*",
    # Error classes (c10 exceptions)
    r"c10::.*Error$",
    r"c10::ErrorAlwaysShowCppStacktrace",
    # Warning internals
    r"c10::Warning.*",
    r"c10::WarningHandler",
    r"c10::WarningUtils::.*",
    # c10 IValue internals
    r"c10::IValue::.*",
    r"c10::IValue",
    r"c10::WeakIValue",
    r"c10::ivalue::.*",
    r"c10::StrongTypePtr",
    r"c10::WeakTypePtr",
    r"c10::WeakOrStrongTypePtr",
    r"c10::WeakOrStrongCompilationUnit",
    r"c10::Capsule",
    r"c10::OptionalArray",
    r"c10::StreamData3",
    # OrderedDict::Item (internal helper)
    r"torch::OrderedDict::Item",
    # ExpandingArray (internal template utility)
    r"torch::ExpandingArray.*",
    # IMethod (internal)
    r"torch::IMethod",
    # CustomClassHolder (internal base)
    r"torch::CustomClassHolder",
    # NodeGuard (internal autograd)
    r"torch::autograd::NodeGuard",
    # Autograd internals
    r"torch::autograd::CppNode",
    r"torch::autograd::ExtractVariables",
    r"torch::autograd::Node",
    r"torch::autograd::Node::.*",
    r"torch::autograd::TraceableFunction",
    r"torch::autograd::TypeAndSize",
    # Sequencer internals
    r"torch::data::.*::detail::.*",
    # cuDNN descriptor internals
    r"at::native::ActivationDescriptor",
    r"at::native::ConvolutionDescriptor",
    r"at::native::SpatialTransformerDescriptor",
    r"at::native::DropoutDescriptor",
    r"at::native::RNNDataDescriptor",
    r"at::native::DftiDescriptor",
    r"at::native::DescriptorDeleter",
    r"at::native::DftiDescriptorDeleter",
    r"at::native::RNNDescriptor",
    # ATen internals
    r"at::OptionalTensorRef",
    r"at::TensorRef",
    # at::cuda internals (allocator, workspace, cublas)
    r"at::cuda::WorkspaceMapWithMutex",
    r"at::cuda::clearCublasWorkspaces.*",
    r"at::cuda::cublas_handle_stream_to_workspace",
    r"at::cuda::cublaslt_handle_stream_to_workspace",
    r"at::cuda::getCUDABlasLt.*",
    r"at::cuda::getCUDADeviceAllocator",
    r"at::cuda::getChosenWorkspaceSize",
    r"at::cuda::getNumGPUs",
    r"at::cuda::is_available",
    r"at::cuda::warp_size",
    # jit namespace (deprecated)
    r"torch::jit::.*",
    # Operators that are just operator<< or operator>>
    r".*::operator<<",
    r".*::operator>>",
    r".*::operator==",
    r".*::operator!=",
    # Internal serialize helpers
    r"torch::optim::serialize",
    r"torch::optim::detail::.*",
    # Reduction enum helpers
    r"torch::nn::reduction",
    r"torch::nn::log_target",
    # Internal module utils
    r"torch::nn::modules::utils::.*",
    # Internal c10 helpers
    r"c10::detail::.*",
    r"c10::detail_::.*",
    r"c10::makeArrayRef",
    r"c10::checkObjectSortSchema",
    r"c10::getGreaterThanComparator",
    r"c10::getLessThanComparator",
    r"c10::value_or_else",
    r"c10::warn",
    r"c10::GetExceptionString",
    # torch::detail
    r"torch::detail::.*",
    # Internal data shuttle/queue
    r"torch::data::detail::.*",
    # DataLoaderBase internal types
    r"torch::data::DataLoaderBase::.*",
    r"torch::data::WorkerException",
    r"torch::data::FullDataLoaderOptions",
    # Template specializations of Stack
    r"torch::data::transforms::Stack< .*>",
    # Example partial specialization
    r"torch::data::Example< .*>",
    # Doxygen internal macros
    r"DEFINE_CASE",
    r"DEFINE_TAG",
    r"COUNT_TAG",
    r"TRUTH_TABLE_ENTRY",
    r"C10_EXPAND_MSVC_WORKAROUND",
    r"TORCH_FORALL_TAGS",
    # Non-public torch::nn functions (module stream operators, etc.)
    r"torch::nn::operator.*",
    # AnyModule/AnyValue internal holders
    r"torch::nn::AnyModuleHolder.*",
    r"torch::nn::AnyModulePlaceholder",
    r"torch::nn::AnyValue.*",
    r"torch::nn::NamedAnyModule",
    # Internal base classes (users use the derived classes)
    r"torch::nn::ConvNdImpl",
    r"torch::nn::ConvTransposeNdImpl",
    r"torch::nn::BatchNormImplBase",
    r"torch::nn::NormImplBase",
    r"torch::nn::InstanceNormImpl",
    r"torch::nn::MaxPoolImpl",
    r"torch::nn::AvgPoolImpl",
    r"torch::nn::AdaptiveAvgPoolImpl",
    r"torch::nn::AdaptiveMaxPoolImpl",
    r"torch::nn::MaxUnpoolImpl",
    r"torch::nn::LPPoolImpl",
    r"torch::nn::ConstantPadImpl",
    r"torch::nn::ReflectionPadImpl",
    r"torch::nn::ReplicationPadImpl",
    r"torch::nn::ZeroPadImpl",
    r"torch::nn::FractionalMaxPoolImpl",
    # nn::functions internal namespace
    r"torch::nn::functions::.*",
    # AdaptiveLogSoftmaxWithLoss (niche, rarely used in C++)
    r"torch::nn::AdaptiveLogSoftmaxWithLoss.*",
    r"torch::nn::ASMoutput",
    # CrossMapLRN2d (niche)
    r"torch::nn::CrossMapLRN2d.*",
    # _out function variants (documented alongside the main function)
    r"torch::special::.*_out",
    r"torch::fft::.*_out",
    # torch internal helpers
    r"torch::InitLambda",
    r"torch::dispatch",
    r"torch::equal_if_defined",
    r"torch::getAllCustomClassesNames",
    r"torch::init",
    r"torch::make_custom_class",
    r"torch::selective_class_",
    r"torch::pickle_load",
    r"torch::pickle_save",
    r"torch::schema",
    r"torch::nativert::.*",
    # RNNCellOptionsBase (internal base)
    r".*::RNNCellOptionsBase",
    # Unnamespaced Options structs (indexed without namespace by Doxygen)
    r"^[A-Z]\w+Options$",
    # Unnamespaced classes without namespace (Doxygen quirk)
    r"^TransformerDecoderLayer$",
    r"^TransformerDecoderLayerOptions$",
    # functional namespace internal options structs
    r"functional::.*FuncOptions",
]

# Specific symbols to exclude (exact match)
EXCLUDED_SYMBOLS = {
    # Internal / not useful to document individually
    "torch::data::datasets::map",
    "torch::data::datasets::make_shared_dataset",
    "torch::data::datasets::operator<<",
    "torch::data::datasets::operator>>",
    "torch::enumtype::get_enum_name",
    "torch::enumtype::reduction_get_enum",
    "torch::autograd::_wrap_outputs",
    "torch::autograd::check_variable_result",
    "torch::autograd::CppNode_apply_functional",
    "torch::autograd::CppNode_apply_functional_ivalue",
    "torch::autograd::forward_ad::enter_dual_level",
    "torch::autograd::forward_ad::exit_dual_level",
    "torch::autograd::any_variable_requires_grad",
    "torch::autograd::collect_next_edges",
    "torch::autograd::create_gradient_edge",
    "torch::autograd::deleteNode",
    "torch::autograd::extract_vars",
    "torch::autograd::get_current_node",
    "torch::autograd::to_optional",
    "torch::autograd::to_output_type",
    "torch::nn::parallel::replicate",
    "torch::nn::parallel::parallel_apply",
    "torch::nn::parallel::data_parallel",
    "torch::python::add_module_bindings",
    "torch::python::bind_module",
    "torch::python::init_bindings",
    # at::native cuDNN internals
    "at::native::dataSize",
    "at::native::fixSizeOneDimStride",
    "at::native::operator<<",
    "at::native::getCudnnDataTypeFromScalarType",
    # c10 cuda pool functions (internal)
    "c10::cuda::getStreamFromPool",
    "c10::cuda::getStreamFromExternal",
    "c10::xpu::getStreamFromPool",
    "c10::xpu::getStreamFromExternal",
    # c10 private use backend registration (internal)
    "c10::get_privateuse1_backend",
    "c10::is_privateuse1_backend_registered",
    "c10::register_privateuse1_backend",
    "c10::isValidDeviceType",
    "c10::DeviceTypeName",
    # torch::stable::detail internals
    "torch::stable::detail::unbox_to_tuple_impl",
    "torch::stable::detail::unbox_to_tuple",
    "torch::stable::detail::box_from_tuple_impl",
    "torch::stable::detail::box_from_tuple",
    # torch::stable::accelerator (documented in stable API page)
    "torch::stable::accelerator::getCurrentStream",
}

# Namespaces whose free functions should be checked for documentation
PUBLIC_FUNCTION_NAMESPACES = {
    "torch",
    "torch::autograd",
    "torch::cuda",
    "torch::mps",
    "torch::xpu",
    "torch::fft",
    "torch::special",
    "torch::nn::functional",
    "torch::nn::init",
    "torch::nn::utils",
    "torch::nn::utils::rnn",
    "torch::data",
    "torch::stable",
    "torch::stable::accelerator",
    "c10",
    "c10::cuda",
    "c10::xpu",
    "at::cuda",
}


# ─── XML parsing ─────────────────────────────────────────────────────────────


def _is_excluded(symbol: str) -> bool:
    """Check if a symbol should be excluded from coverage tracking."""
    if symbol in INCLUDED_SYMBOLS:
        return False
    if symbol in EXCLUDED_SYMBOLS:
        return True
    for pattern in EXCLUDED_PATTERNS:
        if re.fullmatch(pattern, symbol):
            return True
    return False


def _categorize(name: str) -> str:
    """Assign a category based on the symbol's namespace."""
    if name.startswith("torch::nn::functional::"):
        return "torch::nn::functional"
    if name.startswith("torch::nn::init::"):
        return "torch::nn::init"
    if name.startswith("torch::nn::utils::"):
        return "torch::nn::utils"
    if name.startswith("torch::nn::"):
        # Distinguish modules from other nn symbols
        short = name.split("::")[-1]
        if short[0].isupper():
            return "torch::nn (modules)"
        return "torch::nn"
    if name.startswith("torch::optim::"):
        return "torch::optim"
    if name.startswith("torch::data::"):
        return "torch::data"
    if name.startswith("torch::autograd::"):
        return "torch::autograd"
    if name.startswith("torch::serialize::") or name in ("torch::save", "torch::load"):
        return "torch::serialize"
    if name.startswith("torch::stable::"):
        return "torch::stable"
    if name.startswith("torch::fft::"):
        return "torch::fft"
    if name.startswith("torch::special::"):
        return "torch::special"
    if name.startswith(("torch::cuda::", "torch::mps::", "torch::xpu::")):
        return "torch (device)"
    if name.startswith("torch::"):
        return "torch (core)"
    if name.startswith("c10::cuda::"):
        return "c10::cuda"
    if name.startswith("c10::xpu::"):
        return "c10::xpu"
    if name.startswith("c10::"):
        return "c10"
    if name.startswith("at::cuda::"):
        return "at::cuda"
    if name.startswith("at::"):
        return "at"
    return "other"


def discover_apis_from_xml(xml_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Parse Doxygen index.xml to discover all public APIs.

    Returns dict of category -> list of (symbol, kind).
    """
    index_path = xml_dir / "index.xml"
    if not index_path.exists():
        print(
            f"ERROR: {index_path} not found. Run 'make doxygen' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    tree = ET.parse(index_path)
    root = tree.getroot()

    apis: dict[str, list[tuple[str, str]]] = {}

    # Collect classes and structs
    for compound in root.findall("compound"):
        kind = compound.get("kind")
        if kind not in ("class", "struct"):
            continue
        name = compound.find("name").text
        if _is_excluded(name):
            continue
        category = _categorize(name)
        apis.setdefault(category, []).append((name, kind))

    # Collect free functions from public namespaces
    for compound in root.findall("compound"):
        if compound.get("kind") != "namespace":
            continue
        ns_name = compound.find("name").text
        if ns_name not in PUBLIC_FUNCTION_NAMESPACES:
            continue
        seen_funcs = set()
        for member in compound.findall("member"):
            if member.get("kind") != "function":
                continue
            func_name = member.find("name").text
            qualified = f"{ns_name}::{func_name}"
            if qualified in seen_funcs:
                continue  # skip overloads
            seen_funcs.add(qualified)
            if _is_excluded(qualified):
                continue
            category = _categorize(qualified)
            apis.setdefault(category, []).append((qualified, "function"))

    # Collect macros (defines) from file compounds
    for compound in root.findall("compound"):
        if compound.get("kind") != "file":
            continue
        for member in compound.findall("member"):
            if member.get("kind") != "define":
                continue
            macro_name = member.find("name").text
            # Only track well-known public macros
            if macro_name.startswith(("TORCH_LIBRARY", "TORCH_MODULE")):
                if _is_excluded(macro_name):
                    continue
                apis.setdefault("torch (macros)", []).append((macro_name, "define"))

    # Sort each category and deduplicate
    for category in apis:
        apis[category] = sorted(set(apis[category]))

    return apis


# ─── Source scanning ─────────────────────────────────────────────────────────

# RST directives: .. doxygenclass:: torch::nn::ReLU
RST_DIRECTIVE_RE = re.compile(
    r"^\.\.\s+doxygen(class|struct|function|typedef|define|enum|namespace)"
    r"::\s*(.+?)\s*$",
    re.MULTILINE,
)

RST_CPP_DIRECTIVE_RE = re.compile(
    r"^\.\.\s+cpp:(class|struct|function|enum|type)" r"::\s*(.+?)\s*$",
    re.MULTILINE,
)

# MyST directives: ```{doxygenclass} torch::nn::ReLU
MYST_DIRECTIVE_RE = re.compile(
    r"^`{3,}\{doxygen(class|struct|function|typedef|define|enum|namespace)\}\s*(.+?)\s*$",
    re.MULTILINE,
)

MYST_CPP_DIRECTIVE_RE = re.compile(
    r"^`{3,}\{cpp:(class|struct|function|enum|type)\}\s*(.+?)\s*$",
    re.MULTILINE,
)


def scan_sources(source_dir: Path) -> set[str]:
    """Extract all documented symbols from RST/MyST breathe and cpp domain directives."""
    documented = set()
    for src_file in list(source_dir.rglob("*.rst")) + list(source_dir.rglob("*.md")):
        content = src_file.read_text(errors="replace")
        patterns = (
            RST_DIRECTIVE_RE,
            RST_CPP_DIRECTIVE_RE,
            MYST_DIRECTIVE_RE,
            MYST_CPP_DIRECTIVE_RE,
        )
        for pattern in patterns:
            for match in pattern.finditer(content):
                symbol = match.group(2)
                # Strip template prefix
                if symbol.startswith("template"):
                    gt = symbol.find(">")
                    if gt != -1:
                        symbol = symbol[gt + 1 :].lstrip()
                # Strip function signature
                paren = symbol.find("(")
                if paren != -1:
                    symbol = symbol[:paren].rstrip()
                documented.add(symbol)
    return documented


# ─── Coverage report ─────────────────────────────────────────────────────────


def generate_coverage_report(
    apis: dict[str, list[tuple[str, str]]], documented: set[str]
) -> str:
    """Generate a coverage report comparing discovered APIs against RST docs."""
    lines = []
    lines.append("Undocumented C++ objects")
    lines.append("=" * 50)
    lines.append("")

    total = 0
    total_missing = 0
    section_stats = []

    for category in sorted(apis.keys()):
        symbols = apis[category]
        section_missing = []
        for symbol, kind in symbols:
            total += 1
            unqualified = symbol.rsplit("::", 1)[-1]
            if symbol not in documented and unqualified not in documented:
                section_missing.append((symbol, kind))
                total_missing += 1

        covered = len(symbols) - len(section_missing)
        section_stats.append((category, covered, len(symbols)))

        if section_missing:
            lines.append(category)
            lines.append("-" * len(category))
            for symbol, kind in section_missing:
                lines.append(f"   * {symbol}  ({kind})")
            lines.append("")

    # Summary
    total_covered = total - total_missing
    pct = (total_covered / total * 100) if total else 0

    lines.append("")
    lines.append("=" * 50)
    lines.append("Summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Total APIs discovered:   {total}")
    lines.append(f"Documented:              {total_covered}")
    lines.append(f"Missing:                 {total_missing}")
    lines.append(f"Coverage:                {pct:.1f}%")
    lines.append("")

    # Per-section table
    lines.append(f"{'Category':<45} {'Covered':>8} {'Total':>6} {'%':>7}")
    lines.append("-" * 70)
    for category, covered, section_total in section_stats:
        spct = (covered / section_total * 100) if section_total else 0
        lines.append(f"{category:<45} {covered:>8} {section_total:>6} {spct:>6.1f}%")
    lines.append("")

    return "\n".join(lines)


# ─── HTML checks ─────────────────────────────────────────────────────────────

BROKEN_PATTERNS = [
    (
        re.compile(r"Cannot find (?:class|struct|function|file)", re.IGNORECASE),
        "unresolved breathe directive",
    ),
    (
        re.compile(r"Unable to resolve (?:class|struct|function)", re.IGNORECASE),
        "unresolved breathe directive (ambiguous overload)",
    ),
    (
        re.compile(r"doxygenclass:|doxygenfunction:|doxygenstruct:", re.IGNORECASE),
        "raw directive text in output",
    ),
    (
        re.compile(r"<span class=\"problematic\">", re.IGNORECASE),
        "Sphinx problematic node (broken reference)",
    ),
    (
        re.compile(r"System Message:", re.IGNORECASE),
        "Sphinx system message (build error)",
    ),
]

MIN_CONTENT_LENGTH = 500


def check_html_output(build_dir: Path) -> str:
    """Check built HTML for broken formatting and empty pages."""
    issues = []

    if not build_dir.exists():
        return "ERROR: build/html directory not found. Run 'make html' first.\n"

    for html_file in sorted(build_dir.rglob("*.html")):
        rel = html_file.relative_to(build_dir)
        if rel.name in ("search.html", "genindex.html", "objects.inv"):
            continue

        try:
            content = html_file.read_text(errors="replace")
        except Exception as e:
            issues.append((str(rel), f"cannot read: {e}"))
            continue

        for pattern, description in BROKEN_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                issues.append((str(rel), f"{description} ({len(matches)}x)"))

        if str(rel).startswith("api/"):
            text = re.sub(r"<[^>]+>", "", content)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < MIN_CONTENT_LENGTH:
                issues.append((str(rel), f"possibly empty page ({len(text)} chars)"))

    lines = []
    lines.append("HTML Formatting Check")
    lines.append("=" * 50)
    lines.append("")

    if not issues:
        lines.append("No issues found.")
    else:
        lines.append(f"Found {len(issues)} issue(s):")
        lines.append("")
        lines.append(f"{'File':<55} Issue")
        lines.append("-" * 90)
        for filepath, issue in issues:
            lines.append(f"{filepath:<55} {issue}")

    lines.append("")
    return "\n".join(lines)


# ─── coverxygen integration ─────────────────────────────────────────────────


def run_coverxygen(xml_dir: Path) -> str:
    """Run coverxygen on Doxygen XML output for doc-comment coverage."""
    lines = []
    lines.append("Coverxygen Report (Doxygen doc-comment coverage)")
    lines.append("=" * 50)
    lines.append("")

    if not xml_dir.exists():
        lines.append("ERROR: build/xml directory not found. Run 'make doxygen' first.")
        return "\n".join(lines)

    coverxygen_cmd = None
    for cmd in [
        ["coverxygen", "--version"],
        [sys.executable, "-m", "coverxygen", "--version"],
    ]:
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            coverxygen_cmd = cmd[:-1]
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    if coverxygen_cmd is None:
        lines.append("coverxygen not installed. Install with: pip install coverxygen")
        lines.append("")
        lines.append("Once installed, coverxygen analyzes Doxygen XML to report what")
        lines.append("percentage of C++ symbols have doc comments in the source code.")
        lines.append("This is complementary to the RST coverage check above.")
        lines.append("")
        lines.append("Usage:")
        lines.append(
            f"  coverxygen --xml-dir {xml_dir} --src-dir ../../ --output coverxygen.info"
        )
        lines.append("  # Then use lcov/genhtml to visualize:")
        lines.append(
            "  genhtml --no-function-coverage coverxygen.info -o coverxygen_html"
        )
        return "\n".join(lines)

    try:
        result = subprocess.run(
            coverxygen_cmd
            + [
                "--xml-dir",
                str(xml_dir),
                "--src-dir",
                str(SCRIPT_DIR / ".." / ".."),
                "--output",
                "-",
                "--kind",
                "class,struct,function",
                "--scope",
                "public",
                "--exclude",
                ".*/build/.*",
                "--exclude",
                ".*/detail/.*",
                "--exclude",
                ".*/nativert/.*",
                "--exclude",
                ".*/stable/library\\.h",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode == 0:
            total = 0
            documented_count = 0
            for line in result.stdout.splitlines():
                if line.startswith("DA:"):
                    total += 1
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[1].strip() != "0":
                        documented_count += 1
            pct = (documented_count / total * 100) if total else 0
            lines.append(f"Symbols scanned:    {total}")
            lines.append(f"With doc comments:  {documented_count}")
            lines.append(f"Coverage:           {pct:.1f}%")
            lines.append("")
            lines.append("Full lcov output saved to: coverxygen.info")
            (SCRIPT_DIR / "coverxygen.info").write_text(result.stdout)
        else:
            lines.append(f"coverxygen failed (exit {result.returncode}):")
            lines.append(result.stderr[:500])
    except subprocess.TimeoutExpired:
        lines.append("coverxygen timed out after 120s")
    except Exception as e:
        lines.append(f"coverxygen error: {e}")

    lines.append("")
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="C++ docs coverage checker")
    parser.add_argument(
        "--coverxygen",
        action="store_true",
        help="Also run coverxygen on Doxygen XML for doc-comment coverage",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Only run HTML formatting checks",
    )
    args = parser.parse_args()

    reports = []

    if not args.html_only:
        # Phase 1: Discover APIs from Doxygen XML
        print("Discovering APIs from Doxygen XML...")
        apis = discover_apis_from_xml(BUILD_XML)
        total_apis = sum(len(v) for v in apis.values())
        print(f"  Found {total_apis} public APIs across {len(apis)} categories")

        # Phase 2: Scan RST for documented symbols
        print("Scanning sources for breathe directives...")
        documented = scan_sources(SOURCE_DIR)
        print(f"  Found {len(documented)} documented symbols")

        coverage_report = generate_coverage_report(apis, documented)
        reports.append(coverage_report)

        COVERAGE_OUTPUT.write_text(coverage_report)
        print(f"  Coverage report written to: {COVERAGE_OUTPUT}")

    # Phase 3: HTML checks
    print("Checking HTML output for formatting issues...")
    html_report = check_html_output(BUILD_HTML)
    reports.append(html_report)
    HTML_REPORT.write_text(html_report)
    print(f"  HTML report written to: {HTML_REPORT}")

    # Phase 4: coverxygen (optional)
    if args.coverxygen:
        print("Running coverxygen...")
        cov_report = run_coverxygen(BUILD_XML)
        reports.append(cov_report)

    # Print everything
    print()
    print("=" * 60)
    for report in reports:
        print(report)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

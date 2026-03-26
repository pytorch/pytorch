#!/usr/bin/env python3
"""C++ API documentation coverage checker.

Compares a curated allowlist of significant public C++ APIs against
what is actually documented in the RST source files via Breathe directives
(doxygenclass, doxygenstruct, doxygenfunction, etc.).

Outputs a coverage report to cpp_coverage.txt in a format similar to
Sphinx's coverage extension for Python docs.

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
from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "source"
BUILD_HTML = SCRIPT_DIR / "build" / "html"
BUILD_XML = SCRIPT_DIR / "build" / "xml"
COVERAGE_OUTPUT = SCRIPT_DIR / "cpp_coverage.txt"
HTML_REPORT = SCRIPT_DIR / "cpp_html_issues.txt"

# ─── Curated allowlist ───────────────────────────────────────────────────────
# Significant public APIs that should be documented.
# Organized by namespace. Each entry: (symbol, priority, kind)
# kind: "class", "struct", "function", "define", "typedef"
# priority: "high", "medium", "low"

EXPECTED_APIS = {
    "torch::nn (Modules - Activation)": [
        ("torch::nn::ReLU", "high", "class"),
        ("torch::nn::LeakyReLU", "high", "class"),
        ("torch::nn::PReLU", "high", "class"),
        ("torch::nn::RReLU", "medium", "class"),
        ("torch::nn::ReLU6", "medium", "class"),
        ("torch::nn::ELU", "high", "class"),
        ("torch::nn::SELU", "medium", "class"),
        ("torch::nn::CELU", "medium", "class"),
        ("torch::nn::GELU", "high", "class"),
        ("torch::nn::SiLU", "high", "class"),
        ("torch::nn::Mish", "medium", "class"),
        ("torch::nn::GLU", "medium", "class"),
        ("torch::nn::Sigmoid", "high", "class"),
        ("torch::nn::Tanh", "high", "class"),
        ("torch::nn::LogSigmoid", "medium", "class"),
        ("torch::nn::Softmax", "high", "class"),
        ("torch::nn::LogSoftmax", "medium", "class"),
        ("torch::nn::Softmin", "medium", "class"),
        ("torch::nn::Softplus", "medium", "class"),
        ("torch::nn::Softshrink", "low", "class"),
        ("torch::nn::Softsign", "low", "class"),
        ("torch::nn::Hardshrink", "low", "class"),
        ("torch::nn::Hardtanh", "medium", "class"),
        ("torch::nn::Tanhshrink", "low", "class"),
        ("torch::nn::Threshold", "medium", "class"),
    ],
    "torch::nn (Modules - Convolution)": [
        ("torch::nn::Conv1d", "high", "class"),
        ("torch::nn::Conv2d", "high", "class"),
        ("torch::nn::Conv3d", "high", "class"),
        ("torch::nn::ConvTranspose1d", "high", "class"),
        ("torch::nn::ConvTranspose2d", "high", "class"),
        ("torch::nn::ConvTranspose3d", "medium", "class"),
    ],
    "torch::nn (Modules - Pooling)": [
        ("torch::nn::MaxPool1d", "high", "class"),
        ("torch::nn::MaxPool2d", "high", "class"),
        ("torch::nn::MaxPool3d", "medium", "class"),
        ("torch::nn::AvgPool1d", "high", "class"),
        ("torch::nn::AvgPool2d", "high", "class"),
        ("torch::nn::AvgPool3d", "medium", "class"),
        ("torch::nn::AdaptiveAvgPool1d", "medium", "class"),
        ("torch::nn::AdaptiveAvgPool2d", "high", "class"),
        ("torch::nn::AdaptiveAvgPool3d", "medium", "class"),
        ("torch::nn::AdaptiveMaxPool1d", "medium", "class"),
        ("torch::nn::AdaptiveMaxPool2d", "medium", "class"),
        ("torch::nn::AdaptiveMaxPool3d", "low", "class"),
        ("torch::nn::FractionalMaxPool2d", "low", "class"),
        ("torch::nn::FractionalMaxPool3d", "low", "class"),
        ("torch::nn::MaxUnpool1d", "medium", "class"),
        ("torch::nn::MaxUnpool2d", "medium", "class"),
        ("torch::nn::MaxUnpool3d", "medium", "class"),
        ("torch::nn::LPPool1d", "low", "class"),
        ("torch::nn::LPPool2d", "low", "class"),
        ("torch::nn::LPPool3d", "low", "class"),
    ],
    "torch::nn (Modules - Linear)": [
        ("torch::nn::Linear", "high", "class"),
        ("torch::nn::Bilinear", "medium", "class"),
        ("torch::nn::Identity", "high", "class"),
        ("torch::nn::Flatten", "high", "class"),
        ("torch::nn::Unflatten", "medium", "class"),
    ],
    "torch::nn (Modules - Dropout)": [
        ("torch::nn::Dropout", "high", "class"),
        ("torch::nn::Dropout2d", "medium", "class"),
        ("torch::nn::Dropout3d", "medium", "class"),
        ("torch::nn::AlphaDropout", "low", "class"),
        ("torch::nn::FeatureAlphaDropout", "low", "class"),
    ],
    "torch::nn (Modules - Normalization)": [
        ("torch::nn::BatchNorm1d", "high", "class"),
        ("torch::nn::BatchNorm2d", "high", "class"),
        ("torch::nn::BatchNorm3d", "medium", "class"),
        ("torch::nn::InstanceNorm1d", "medium", "class"),
        ("torch::nn::InstanceNorm2d", "medium", "class"),
        ("torch::nn::InstanceNorm3d", "low", "class"),
        ("torch::nn::LayerNorm", "high", "class"),
        ("torch::nn::GroupNorm", "high", "class"),
        ("torch::nn::LocalResponseNorm", "low", "class"),
    ],
    "torch::nn (Modules - Embedding)": [
        ("torch::nn::Embedding", "high", "class"),
        ("torch::nn::EmbeddingBag", "medium", "class"),
    ],
    "torch::nn (Modules - Recurrent)": [
        ("torch::nn::RNN", "high", "class"),
        ("torch::nn::LSTM", "high", "class"),
        ("torch::nn::GRU", "high", "class"),
        ("torch::nn::RNNCell", "medium", "class"),
        ("torch::nn::LSTMCell", "medium", "class"),
        ("torch::nn::GRUCell", "medium", "class"),
    ],
    "torch::nn (Modules - Transformer)": [
        ("torch::nn::Transformer", "high", "class"),
        ("torch::nn::TransformerEncoder", "high", "class"),
        ("torch::nn::TransformerDecoder", "high", "class"),
        ("torch::nn::TransformerEncoderLayerImpl", "medium", "class"),
        ("torch::nn::TransformerDecoderLayerImpl", "medium", "class"),
        ("torch::nn::MultiheadAttention", "high", "class"),
    ],
    "torch::nn (Modules - Loss)": [
        ("torch::nn::L1Loss", "high", "class"),
        ("torch::nn::MSELoss", "high", "class"),
        ("torch::nn::CrossEntropyLoss", "high", "class"),
        ("torch::nn::NLLLoss", "high", "class"),
        ("torch::nn::BCELoss", "high", "class"),
        ("torch::nn::BCEWithLogitsLoss", "high", "class"),
        ("torch::nn::HuberLoss", "medium", "class"),
        ("torch::nn::SmoothL1Loss", "medium", "class"),
        ("torch::nn::KLDivLoss", "medium", "class"),
        ("torch::nn::CTCLoss", "medium", "class"),
        ("torch::nn::PoissonNLLLoss", "low", "class"),
        ("torch::nn::MarginRankingLoss", "low", "class"),
        ("torch::nn::HingeEmbeddingLoss", "low", "class"),
        ("torch::nn::CosineEmbeddingLoss", "low", "class"),
        ("torch::nn::MultiMarginLoss", "low", "class"),
        ("torch::nn::MultiLabelMarginLoss", "low", "class"),
        ("torch::nn::MultiLabelSoftMarginLoss", "low", "class"),
        ("torch::nn::SoftMarginLoss", "low", "class"),
        ("torch::nn::TripletMarginLoss", "medium", "class"),
        ("torch::nn::TripletMarginWithDistanceLoss", "low", "class"),
    ],
    "torch::nn (Modules - Containers)": [
        ("torch::nn::Sequential", "high", "class"),
        ("torch::nn::ModuleList", "high", "class"),
        ("torch::nn::ModuleDict", "medium", "class"),
        ("torch::nn::ParameterList", "medium", "class"),
        ("torch::nn::ParameterDict", "medium", "class"),
    ],
    "torch::nn (Modules - Utilities)": [
        ("torch::nn::Module", "high", "class"),
        ("torch::nn::Cloneable", "medium", "class"),
        ("torch::nn::AnyModule", "medium", "class"),
        ("torch::nn::Functional", "medium", "class"),
        ("torch::nn::ModuleHolder", "medium", "class"),
        ("torch::nn::CosineSimilarity", "medium", "class"),
        ("torch::nn::PairwiseDistance", "medium", "class"),
        ("torch::nn::utils::rnn::PackedSequence", "medium", "class"),
    ],
    "torch::nn (Modules - Padding/Vision)": [
        ("torch::nn::ReflectionPad1d", "medium", "class"),
        ("torch::nn::ReflectionPad2d", "medium", "class"),
        ("torch::nn::ReflectionPad3d", "low", "class"),
        ("torch::nn::ReplicationPad1d", "medium", "class"),
        ("torch::nn::ReplicationPad2d", "medium", "class"),
        ("torch::nn::ReplicationPad3d", "low", "class"),
        ("torch::nn::ZeroPad1d", "medium", "class"),
        ("torch::nn::ZeroPad2d", "medium", "class"),
        ("torch::nn::ZeroPad3d", "low", "class"),
        ("torch::nn::ConstantPad1d", "medium", "class"),
        ("torch::nn::ConstantPad2d", "medium", "class"),
        ("torch::nn::ConstantPad3d", "low", "class"),
        ("torch::nn::PixelShuffle", "medium", "class"),
        ("torch::nn::PixelUnshuffle", "medium", "class"),
        ("torch::nn::Upsample", "medium", "class"),
        ("torch::nn::Fold", "low", "class"),
        ("torch::nn::Unfold", "low", "class"),
    ],
    "torch::nn::functional": [
        ("torch::nn::functional::relu", "high", "function"),
        ("torch::nn::functional::leaky_relu", "high", "function"),
        ("torch::nn::functional::elu", "medium", "function"),
        ("torch::nn::functional::selu", "medium", "function"),
        ("torch::nn::functional::gelu", "high", "function"),
        ("torch::nn::functional::silu", "high", "function"),
        ("torch::nn::functional::mish", "medium", "function"),
        ("torch::nn::functional::softmax", "high", "function"),
        ("torch::nn::functional::log_softmax", "medium", "function"),
        ("torch::nn::functional::linear", "high", "function"),
        ("torch::nn::functional::conv1d", "high", "function"),
        ("torch::nn::functional::conv2d", "high", "function"),
        ("torch::nn::functional::conv3d", "medium", "function"),
        ("torch::nn::functional::max_pool2d", "high", "function"),
        ("torch::nn::functional::avg_pool2d", "high", "function"),
        ("torch::nn::functional::dropout", "high", "function"),
        ("torch::nn::functional::batch_norm", "high", "function"),
        ("torch::nn::functional::layer_norm", "high", "function"),
        ("torch::nn::functional::cross_entropy", "high", "function"),
        ("torch::nn::functional::nll_loss", "medium", "function"),
        ("torch::nn::functional::mse_loss", "medium", "function"),
        ("torch::nn::functional::l1_loss", "medium", "function"),
        ("torch::nn::functional::embedding", "medium", "function"),
        ("torch::nn::functional::interpolate", "medium", "function"),
        ("torch::nn::functional::pad", "medium", "function"),
        ("torch::nn::functional::normalize", "medium", "function"),
        ("torch::nn::functional::one_hot", "medium", "function"),
        ("torch::nn::functional::cosine_similarity", "medium", "function"),
        ("torch::nn::functional::pairwise_distance", "medium", "function"),
    ],
    "torch::optim": [
        ("torch::optim::Optimizer", "high", "class"),
        ("torch::optim::SGD", "high", "class"),
        ("torch::optim::Adam", "high", "class"),
        ("torch::optim::AdamW", "high", "class"),
        ("torch::optim::RMSprop", "medium", "class"),
        ("torch::optim::Adagrad", "medium", "class"),
        ("torch::optim::LBFGS", "medium", "class"),
        ("torch::optim::LRScheduler", "high", "class"),
        ("torch::optim::StepLR", "high", "class"),
        ("torch::optim::ReduceLROnPlateauScheduler", "high", "class"),
        ("torch::optim::OptimizerOptions", "medium", "class"),
        ("torch::optim::OptimizerParamGroup", "medium", "class"),
        ("torch::optim::OptimizerParamState", "medium", "class"),
    ],
    "torch::data": [
        ("torch::data::DataLoaderBase", "high", "class"),
        ("torch::data::DataLoaderOptions", "medium", "struct"),
        ("torch::data::Example", "medium", "struct"),
        ("torch::data::Iterator", "medium", "class"),
        ("torch::data::datasets::Dataset", "high", "class"),
        ("torch::data::datasets::BatchDataset", "medium", "class"),
        ("torch::data::datasets::MNIST", "medium", "class"),
        ("torch::data::datasets::MapDataset", "medium", "class"),
        ("torch::data::datasets::ChunkDataset", "low", "class"),
        ("torch::data::datasets::SharedBatchDataset", "low", "class"),
        ("torch::data::samplers::Sampler", "high", "class"),
        ("torch::data::samplers::SequentialSampler", "medium", "class"),
        ("torch::data::samplers::RandomSampler", "medium", "class"),
        ("torch::data::samplers::DistributedRandomSampler", "medium", "class"),
        ("torch::data::samplers::DistributedSampler", "medium", "class"),
        ("torch::data::samplers::DistributedSequentialSampler", "medium", "class"),
        ("torch::data::samplers::StreamSampler", "low", "class"),
        ("torch::data::transforms::Transform", "medium", "class"),
        ("torch::data::transforms::BatchTransform", "low", "class"),
        ("torch::data::transforms::TensorTransform", "low", "class"),
        ("torch::data::transforms::Normalize", "medium", "struct"),
        ("torch::data::transforms::Stack", "medium", "struct"),
        ("torch::data::transforms::Lambda", "low", "class"),
        ("torch::data::transforms::TensorLambda", "low", "class"),
        ("torch::data::transforms::BatchLambda", "low", "class"),
    ],
    "torch::autograd": [
        ("torch::autograd::Function", "high", "struct"),
        ("torch::autograd::AutogradContext", "high", "struct"),
        ("torch::autograd::grad", "high", "function"),
    ],
    "torch::serialize": [
        ("torch::serialize::OutputArchive", "high", "class"),
        ("torch::serialize::InputArchive", "high", "class"),
        ("torch::save", "high", "function"),
        ("torch::load", "high", "function"),
    ],
    "torch (Library/Registration)": [
        ("torch::Library", "high", "class"),
        ("torch::class_", "medium", "class"),
        ("TORCH_LIBRARY", "high", "define"),
        ("TORCH_LIBRARY_IMPL", "high", "define"),
        ("TORCH_LIBRARY_FRAGMENT", "medium", "define"),
        ("torch::CppFunction", "medium", "class"),
        ("torch::OrderedDict", "medium", "class"),
    ],
    "torch::stable": [
        ("torch::stable::Tensor", "high", "class"),
        ("torch::stable::Device", "medium", "class"),
        ("torch::stable::accelerator::DeviceGuard", "medium", "class"),
        ("torch::stable::accelerator::Stream", "medium", "class"),
        ("torch::stable::empty", "medium", "function"),
        ("torch::stable::from_blob", "medium", "function"),
        ("torch::stable::matmul", "medium", "function"),
        ("torch::stable::accelerator::getCurrentDeviceIndex", "medium", "function"),
        ("torch::stable::parallel_for", "medium", "function"),
    ],
    "c10 (Core)": [
        ("c10::Device", "high", "struct"),
        ("c10::DeviceGuard", "high", "class"),
        ("c10::OptionalDeviceGuard", "high", "class"),
        ("c10::Stream", "high", "class"),
        ("c10::ArrayRef", "high", "class"),
        ("c10::OptionalArrayRef", "medium", "class"),
        ("c10::Dict", "medium", "class"),
        ("c10::List", "medium", "class"),
    ],
    "CUDA": [
        ("c10::cuda::CUDAStream", "high", "class"),
        ("c10::cuda::CUDAGuard", "high", "struct"),
        ("c10::cuda::CUDAStreamGuard", "medium", "struct"),
        ("c10::cuda::OptionalCUDAGuard", "medium", "struct"),
        ("c10::cuda::OptionalCUDAStreamGuard", "medium", "struct"),
        ("c10::cuda::CUDAMultiStreamGuard", "medium", "struct"),
        ("c10::cuda::getDefaultCUDAStream", "high", "function"),
        ("c10::cuda::getCurrentCUDAStream", "high", "function"),
        ("c10::cuda::setCurrentCUDAStream", "medium", "function"),
        ("at::cuda::getCurrentDeviceProperties", "medium", "function"),
    ],
    "XPU": [
        ("c10::xpu::XPUStream", "medium", "class"),
        ("c10::xpu::getCurrentXPUStream", "medium", "function"),
        ("c10::xpu::setCurrentXPUStream", "medium", "function"),
        ("torch::xpu::device_count", "medium", "function"),
        ("torch::xpu::is_available", "medium", "function"),
        ("torch::xpu::synchronize", "medium", "function"),
    ],
    "ATen": [
        ("at::Tensor", "high", "class"),
        ("at::native::Descriptor", "low", "class"),
        ("at::native::TensorDescriptor", "low", "class"),
        ("at::native::FilterDescriptor", "low", "class"),
    ],
}


# ─── RST scanning ────────────────────────────────────────────────────────────

# Matches breathe directives: .. doxygenclass:: torch::nn::ReLU
DIRECTIVE_RE = re.compile(
    r"^\.\.\s+doxygen(class|struct|function|typedef|define|enum|namespace)"
    r"::\s*(.+?)\s*$",
    re.MULTILINE,
)

# Matches manual Sphinx C++ domain directives: .. cpp:class:: at::Tensor
CPP_DIRECTIVE_RE = re.compile(
    r"^\.\.\s+cpp:(class|struct|function|enum|type)" r"::\s*(.+?)\s*$",
    re.MULTILINE,
)


def scan_rst_sources(source_dir: Path) -> set[str]:
    """Extract all documented symbols from RST breathe and cpp domain directives."""
    documented = set()
    for rst_file in source_dir.rglob("*.rst"):
        content = rst_file.read_text(errors="replace")
        for pattern in (DIRECTIVE_RE, CPP_DIRECTIVE_RE):
            for match in pattern.finditer(content):
                symbol = match.group(2)
                # Strip template prefix: "template<...> c10::Dict" → "c10::Dict"
                if symbol.startswith("template"):
                    gt = symbol.find(">")
                    if gt != -1:
                        symbol = symbol[gt + 1 :].lstrip()
                # Strip function signature if present: "torch::save(...)" → "torch::save"
                paren = symbol.find("(")
                if paren != -1:
                    symbol = symbol[:paren].rstrip()
                documented.add(symbol)
    return documented


# ─── Coverage report ─────────────────────────────────────────────────────────


def generate_coverage_report(documented: set[str]) -> str:
    """Generate a coverage report in the style of sphinx.ext.coverage."""
    lines = []
    lines.append("Undocumented C++ objects")
    lines.append("=" * 50)
    lines.append("")

    total = 0
    total_missing = 0
    missing_by_priority = {"high": [], "medium": [], "low": []}
    section_stats = []

    for section, apis in EXPECTED_APIS.items():
        section_missing = []
        for symbol, priority, kind in apis:
            total += 1
            # Check fully-qualified name or unqualified name (some Doxygen
            # entries lack namespace, e.g. TransformerDecoderLayerImpl)
            unqualified = symbol.rsplit("::", 1)[-1]
            if symbol not in documented and unqualified not in documented:
                section_missing.append((symbol, priority, kind))
                total_missing += 1
                missing_by_priority[priority].append(symbol)

        covered = len(apis) - len(section_missing)
        section_stats.append((section, covered, len(apis)))

        if section_missing:
            lines.append(section)
            lines.append("-" * len(section))
            for symbol, priority, kind in section_missing:
                lines.append(f"   * {symbol}  ({kind}, {priority})")
            lines.append("")

    # Summary
    total_covered = total - total_missing
    pct = (total_covered / total * 100) if total else 0

    lines.append("")
    lines.append("=" * 50)
    lines.append("Summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Total APIs in allowlist:  {total}")
    lines.append(f"Documented:              {total_covered}")
    lines.append(f"Missing:                 {total_missing}")
    lines.append(f"Coverage:                {pct:.1f}%")
    lines.append("")

    # Per-section table
    lines.append(f"{'Section':<45} {'Covered':>8} {'Total':>6} {'%':>7}")
    lines.append("-" * 70)
    for section, covered, section_total in section_stats:
        spct = (covered / section_total * 100) if section_total else 0
        lines.append(f"{section:<45} {covered:>8} {section_total:>6} {spct:>6.1f}%")
    lines.append("")

    # Priority breakdown
    lines.append("Missing by priority:")
    for p in ("high", "medium", "low"):
        syms = missing_by_priority[p]
        lines.append(f"  {p:<8}: {len(syms)}")
        for s in syms:
            lines.append(f"             - {s}")
    lines.append("")

    return "\n".join(lines)


# ─── HTML checks ─────────────────────────────────────────────────────────────

# Patterns indicating broken rendering
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
    (re.compile(r"System Message:", re.IGNORECASE), "Sphinx system message (build error)"),
]

# Minimum content length for a page to not be considered empty
MIN_CONTENT_LENGTH = 500


def check_html_output(build_dir: Path) -> str:
    """Check built HTML for broken formatting and empty pages."""
    issues = []

    if not build_dir.exists():
        return "ERROR: build/html directory not found. Run 'make html' first.\n"

    for html_file in sorted(build_dir.rglob("*.html")):
        rel = html_file.relative_to(build_dir)
        # Skip search and genindex
        if rel.name in ("search.html", "genindex.html", "objects.inv"):
            continue

        try:
            content = html_file.read_text(errors="replace")
        except Exception as e:
            issues.append((str(rel), f"cannot read: {e}"))
            continue

        # Check for broken patterns
        for pattern, description in BROKEN_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                issues.append((str(rel), f"{description} ({len(matches)}x)"))

        # Check for near-empty API pages (only check api/ subdirectory)
        if str(rel).startswith("api/"):
            # Strip HTML tags for content length check
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

    # Try CLI first, then fall back to python -m
    coverxygen_cmd = None
    for cmd in [
        ["coverxygen", "--version"],
        [sys.executable, "-m", "coverxygen", "--version"],
    ]:
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            coverxygen_cmd = cmd[:-1]  # strip --version
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

    # Run coverxygen with summary
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
                # Exclude auto-generated code and internal implementation details.
                # To exclude additional paths, add more --exclude entries below.
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
        )
        if result.returncode == 0:
            # Count documented vs total from lcov-style output
            total = 0
            documented = 0
            for line in result.stdout.splitlines():
                if line.startswith("DA:"):
                    total += 1
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[1].strip() != "0":
                        documented += 1
            pct = (documented / total * 100) if total else 0
            lines.append(f"Symbols scanned:    {total}")
            lines.append(f"With doc comments:  {documented}")
            lines.append(f"Coverage:           {pct:.1f}%")
            lines.append("")
            lines.append("Full lcov output saved to: coverxygen.info")
            # Save full output
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
        # Phase 1: RST coverage
        print("Scanning RST sources for breathe directives...")
        documented = scan_rst_sources(SOURCE_DIR)
        print(f"  Found {len(documented)} documented symbols")

        coverage_report = generate_coverage_report(documented)
        reports.append(coverage_report)

        # Write coverage output
        COVERAGE_OUTPUT.write_text(coverage_report)
        print(f"  Coverage report written to: {COVERAGE_OUTPUT}")

    # Phase 2: HTML checks
    print("Checking HTML output for formatting issues...")
    html_report = check_html_output(BUILD_HTML)
    reports.append(html_report)
    HTML_REPORT.write_text(html_report)
    print(f"  HTML report written to: {HTML_REPORT}")

    # Phase 3: coverxygen (optional)
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

    # Exit with error if there are missing high-priority APIs
    if not args.html_only:
        documented = scan_rst_sources(SOURCE_DIR)
        high_missing = 0
        for apis in EXPECTED_APIS.values():
            for symbol, priority, kind in apis:
                if priority == "high" and symbol not in documented:
                    high_missing += 1
        if high_missing > 0:
            print(f"FAIL: {high_missing} high-priority APIs missing documentation")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

# We hook directly into include-what-you-use's fix_includes.py tool
# because it doesn't allow generating just a diff without modifying the file
from .fix_includes import (
    FixIncludesError,
    FileInfo,
    NormalizeFilePath,
    IWYUOutputParser,
    IWYUOutputRecord,
    ParseOneFile,
    FixFileLines,
)
from .lint_message import LintMessage, LintSeverity
from .fixup import (
    use_angled_includes,
    normalize_c_headers,
    use_quotes_for_project_includes,
)

from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Sequence,
    Optional,
    Iterable,
    TypeVar,
    Iterator,
    List,
    Dict,
)
from collections import OrderedDict
import logging
import io


T = TypeVar("T")
U = TypeVar("U")


def concatMap(func: Callable[[T], Sequence[U]], xs: Iterable[T]) -> Iterator[U]:
    for x in xs:
        yield from func(x)


@dataclass(frozen=True)
class FixIncludeFlags:
    blank_lines: bool = True
    comments: bool = True
    update_comments: bool = False
    safe_headers: bool = True
    reorder: bool = False
    sort_only: bool = False
    separate_project_includes: Optional[str] = None
    keep_iwyu_namespace_format: bool = False
    basedir: Optional[str] = None


def process_iwyu_output(input: str, flags: FixIncludeFlags) -> List[LintMessage]:
    """Fix the #include and forward-declare lines as directed by f.

    Given a file object that has the output of the include_what_you_use
    script, see every file to be edited and edit it, if appropriate.

    Returns:
      The number of files that had to be modified (because they weren't
      already all correct).  In dry_run mode, returns the number of
      files that would have been modified.
    """
    # First collect all the iwyu data from stdin.
    input_lines = []
    for line in input.splitlines():
        line = use_angled_includes(line)
        line = normalize_c_headers(line)
        line = use_quotes_for_project_includes(line)
        input_lines.append(line)

    input_stream = io.StringIO("\n".join(input_lines))

    # Maintain sort order by using OrderedDict instead of dict
    iwyu_output_records: Dict[str, IWYUOutputRecord] = OrderedDict()
    errors: List[LintMessage] = []
    while True:
        iwyu_output_parser = IWYUOutputParser()
        try:
            iwyu_record = iwyu_output_parser.ParseOneRecord(input_stream, flags)
            if not iwyu_record:
                break
        except FixIncludesError as why:
            errors.append(
                LintMessage(
                    path="",
                    line=None,
                    char=None,
                    code="IWYU",
                    severity=LintSeverity.ADVICE,
                    name="command-failed",
                    original=None,
                    replacement=None,
                    description=f"Parseing iwyu output failed:\n{why}",
                )
            )
            continue
        filename = NormalizeFilePath(flags.basedir, iwyu_record.filename)

        if filename in iwyu_output_records:
            iwyu_output_records[filename].Merge(iwyu_record)
        else:
            iwyu_output_records[filename] = iwyu_record

    # Now ignore all the files that never had any contentful changes
    # seen for them.  (We have to wait until we're all done, since a .h
    # file may have a contentful change when #included from one .cc
    # file, but not another, and we need to have merged them above.)
    if not flags.update_comments:

        def skip_output_record(record: IWYUOutputRecord) -> bool:
            skip = not record.HasContentfulChanges()
            if skip:
                logging.debug(
                    f"skipping {filename}: iwyu reports no contentful changes"
                )
            return skip

        iwyu_output_records = {
            filename: record
            for filename, record in iwyu_output_records.items()
            if not skip_output_record(record)
        }

    # Now do all the fixing, and return the number of files modified
    contentful_records = [ior for ior in iwyu_output_records.values() if ior]
    diffs = list(concatMap(partial(generate_diff, flags=flags), contentful_records))
    return errors + diffs


def _ReadFile(filename: str, fileinfo: FileInfo) -> Optional[List[str]]:
    """Read from filename and return a list of file lines."""
    with open(filename, "rb") as f:
        content = f.read()
        # Call splitlines with True to keep the original line
        # endings.  Later in WriteFile, they will be used as-is.
        # This will reduce spurious changes to the original files.
        # The lines we add will have the linesep determined by
        # FileInfo.
        return content.decode(fileinfo.encoding).splitlines(True)


def generate_diff(
    iwyu_record: IWYUOutputRecord, flags: FixIncludeFlags
) -> List[LintMessage]:
    try:
        fileinfo = FileInfo.parse(iwyu_record.filename)

        original_file_contents = _ReadFile(iwyu_record.filename, fileinfo)
        if not original_file_contents:
            return []

        # IWYU expects project includes to always use "quoted includes"
        file_lines = ParseOneFile(
            [use_quotes_for_project_includes(line) for line in original_file_contents],
            iwyu_record,
        )

        old_lines = [
            fl.line for fl in file_lines if fl is not None and fl.line is not None
        ]
        fixed_lines = FixFileLines(iwyu_record, file_lines, flags, fileinfo)

        # Convert IWYU's quoted includes to ATen-style angled includes
        fixed_lines = [use_angled_includes(line) for line in fixed_lines]

        return [
            LintMessage(
                path=iwyu_record.filename,
                line=None,
                char=None,
                code="IWYU",
                severity=LintSeverity.WARNING,
                name="include-what-you-use",
                original="".join(original_file_contents),
                replacement="".join(fixed_lines),
                description="Fix includes",
            )
        ]
    except FixIncludesError as why:
        return [
            LintMessage(
                path=iwyu_record.filename,
                line=None,
                char=None,
                code="IWYU",
                severity=LintSeverity.ADVICE,
                name="command-failed",
                original=None,
                replacement=None,
                description=f"{why} - skipping file {iwyu_record.filename}",
            ),
        ]

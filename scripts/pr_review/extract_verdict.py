#!/usr/bin/env python3
"""Sanitize the hardened PR review's model output into a publishable verdict.

The review job runs Claude against UNTRUSTED pull-request content with network
egress locked down, no GitHub write token, and no S3 credentials. The only thing
that leaves that job is the JSON this script produces, so this file is the egress
boundary and every control here exists because that text is attacker-influenced
and ends up somewhere a human or a dashboard reads.

What a JSON schema already guarantees (the action is invoked with --json-schema):
shape, types, the verdict enum, required keys. Those guarantees come from an
attacker-influenced producer, so this script re-checks them rather than assuming
them, and adds what a schema cannot express:

  * charset — every published STRING must be printable ASCII plus tab and
    newline. Checked on the decoded values, not on the JSON serialization:
    `json.dumps` renders an ESC byte as the printable text `\\u001b`, so
    validating the serialized form would pass a control sequence straight
    through to whatever decodes it.
  * encoded blobs — a long base64/base32/hex run that decodes to mostly
    non-printable bytes is treated as smuggled binary, not as prose.
  * anchoring — every finding must point at a file the PR actually changed, and
    at a line the diff actually touches. This is the real bandwidth reducer: it
    turns N free-text findings into "must be attached to a real changed line".
  * rendering safety — @mentions, #1234 cross-references, links, images and HTML
    are neutralized so a finding cannot notify maintainers, phish, or beacon.
  * caps — bounded finding count and bounded string lengths, applied to EVERY
    string that reaches the output, including diagnostic ones.

None of this closes the channel, and it is not the thing keeping us safe. A few
hundred characters of prose can still carry a secret via an acrostic or semantic
encoding, and no output filter fixes that. The control that actually bounds the
damage is upstream: a 15-minute, Bedrock-only role with no S3 and no GitHub
write. This script raises the cost of the naive attempts and makes the
deliberate ones narrow.

Exit code is always 0, and a well-formed file is always written — a rejected or
malformed result is reported as a `blocked`/`schema_invalid` status rather than
as a failed job, so the pipeline records what happened instead of going silent.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import codecs
import json
import re
import sys
from pathlib import Path


VERDICTS = {"ready_for_human_review", "changes_requested"}
SEVERITIES = {"info", "minor", "major"}

MAX_FINDINGS = 25
MAX_MESSAGE = 600
MAX_SUMMARY = 1500
MAX_PATH = 400
MAX_DROPPED_TRACKED = (
    200  # bound the diagnostic list; the model can emit unlimited findings
)
MAX_STRUCTURED_BYTES = 1024 * 1024  # whole-document ceiling, applied before parsing

# Printable ASCII plus tab and newline. Everything else — control characters,
# zero-width joiners, variation selectors, RTL overrides, any non-ASCII — fails.
_ALLOWED_CHARS = re.compile(r"^[\x09\x0a\x20-\x7e]*$")

# Stricter than _ALLOWED_CHARS: no tab, no newline. Applied to diff paths only.
_PATH_CHARS = re.compile(r"^[\x20-\x7e]+$")

# Long unbroken runs that look like an encoded payload rather than prose.
#
# The 120-character floor is calibrated, not arbitrary. Shorter runs are
# dominated by things a real review says: a 40-char git SHA, a 64-char sha256,
# a long CONSTANT_NAME. Those are decodable and land on high-entropy bytes, so a
# lower floor rejects legitimate reviews wholesale: at 40 characters, "regressed
# in <sha>" is refused. What we care about hiding is a credential, and an AWS
# session token is ~700-1000 base64 characters.
_BLOB = re.compile(r"[A-Za-z0-9+/=_-]{120,}")

# Hex digests are valid base64 syntax and decode to high-entropy bytes, so
# without this they trip the blob guard and reject the whole review. A SHA-512
# hex digest is 128 characters, which is over the floor.
# ACCEPTED COST: this leaves a hole exactly the width of a digest — an attacker
# can hex-encode up to 64 bytes and shape it like one. That is far below a
# session token, and the 15-minute Bedrock-only credential is what bounds the
# damage anyway.
_DIGEST_LENGTHS = {32, 40, 56, 64, 96, 128}
_HEX_ONLY = re.compile(r"^[0-9a-fA-F]+$")

# Rendering hazards. Neutralized (not rejected) because they occur innocently in
# real review prose — a finding may legitimately mention `user@example.com`.
#
# The lookbehind is `\w`, NOT `[\w/]`. Exempting `/` was meant to spare paths,
# but `/` is precisely the character in front of the two forms that DO notify:
# GitHub linkifies `@name` and `#123` after any non-word character, and
# `owner/repo#123` is the cross-repository reference — it notifies AND writes a
# cross-reference event into the target issue. `torch/@pytorch-dev-infra` and
# `pytorch/pytorch#12345` both survived the old pattern untouched.
_MENTION = re.compile(r"(?<!\w)@([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))")
_ISSUE_REF = re.compile(r"(?<!\w)#(\d+)")
# `owner/repo#123` has a word character before the `#`, so _ISSUE_REF cannot see
# it however the lookbehind is written. It needs its own pattern, run first.
_XREF = re.compile(r"(?<![\w/])([A-Za-z0-9._-]{1,64}/[A-Za-z0-9._-]{1,64})#(\d+)")
# Scheme-relative `//host/path` is a live link in both HTML and markdown and was
# not covered: `_URL` required `scheme://`, so `[click](//evil.example.com)`  # @lint-ignore
# kept its target through every pass. `mailto:` and `tel:` have no `//` at all.
#
# The host alternative demands a dot BEFORE the first slash, which is what keeps
# `//caffe2/core:core` (a Buck label, plausible in review prose) and `// TODO`
# out of it while still matching `//user@evil.example.com/` and `//203.0.113.9/x`.  # @lint-ignore
_URL = re.compile(
    r"(?:\b(?:https?|ftp|data|javascript|vbscript|file)://\S+"
    r"|\b(?:mailto|tel):\S+"
    r"|(?<![:\w/])//(?:[^\s/@]{1,64}@)?[A-Za-z0-9._~-]*\.[A-Za-z0-9._~-]+\S*)",
    re.IGNORECASE,
)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Replacement markers use PARENTHESES, not brackets. With brackets they were
# themselves usable as a reference-link label: `[link removed]: //evil` followed
# by `Click [link removed] to continue.` produced a live link wearing the
# sanitizer's own text.
_LINK_MARKER = "(link removed)"
_IMAGE_MARKER = "(image removed)"

# Was `<[^>\n]{1,200}>`, and both bounds were holes. Excluding newline meant a
# tag split across lines was invisible to it, and `</details\n>` is a VALID end
# tag — whitespace before `>` is legal — so it survived and closed the
# renderer's own collapsible block, promoting attacker text into the comment's
# top-level framing. The 200-character cap was the second hole: any longer
# attribute list walked through untouched. `[^>]` already spans newlines in
# Python, so dropping both bounds is the whole fix.
_HTML_TAG = re.compile(r"<[^>]*>")

# Applied AFTER tag stripping, to whatever `<`, `>` or `&` is left over — an
# unterminated `<!--`, say, which has no closing `>` to match on and would
# otherwise swallow everything a renderer shows after it. Escaping cannot be applied
# first: a tag that VANISHES is what lets `https:<i></i>//evil` reassemble, and
# the blob check downstream is calibrated to see that reassembly happen.
_HTML_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"))


class Rejected(Exception):
    """The output cannot be published at all."""


def iter_strings(obj):
    """Yield every string reachable in a decoded JSON value, keys included."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str):
                yield key
            yield from iter_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_strings(item)


def check_charset(text: str, where: str) -> None:
    if not _ALLOWED_CHARS.match(text):
        bad = {hex(ord(c)) for c in text if not _ALLOWED_CHARS.match(c)}
        raise Rejected(
            f"{where}: contains {len(bad)} disallowed codepoint kind(s) "
            f"(e.g. {sorted(bad)[:5]}); only printable ASCII, tab and newline may be published"
        )


def check_value_charset(obj, where: str) -> None:
    """Charset-check the DECODED strings of a JSON value.

    Checking `json.dumps(obj)` instead would be a false guarantee: the escape
    `\\u001b` is printable ASCII in the serialized text while the value it
    denotes is an ESC byte.
    """
    for text in iter_strings(obj):
        check_charset(text, where)


def _looks_like_digest(blob: str) -> bool:
    return len(blob) in _DIGEST_LENGTHS and bool(_HEX_ONLY.match(blob))


def _decodes_to_binary(blob: str) -> bool:
    """True if `blob` decodes to bytes that are mostly non-printable.

    Each codec gets the padding IT needs: base64 pads to a multiple of 4,
    base32 to a multiple of 8, and hex takes none at all. Padding everything to
    4 makes most valid base32 and odd-length hex runs fail to decode, which
    silently stops the guard inspecting them — do not collapse these branches.
    """
    if _looks_like_digest(blob):
        return False
    candidates = (
        (base64.b64decode, blob + "=" * (-len(blob) % 4), {"validate": True}),
        (base64.urlsafe_b64decode, blob + "=" * (-len(blob) % 4), {}),
        (base64.b32decode, blob + "=" * (-len(blob) % 8), {"casefold": True}),
        (binascii.unhexlify, blob, {}),
    )
    for decoder, payload, kwargs in candidates:
        try:
            raw = decoder(payload, **kwargs)
        except (binascii.Error, ValueError, TypeError):
            continue
        if len(raw) < 24:
            continue
        printable = sum(1 for b in raw if 0x20 <= b <= 0x7E or b in (0x09, 0x0A, 0x0D))
        if printable / len(raw) < 0.75:
            return True
    return False


def check_no_encoded_blob(text: str, where: str) -> None:
    """Reject long encoding-shaped runs that decode to binary.

    HONEST SCOPE: defence in depth, not a data-loss boundary, and bypassable —
    splitting a payload into sub-120-character runs, or base64ing text that
    decodes back to printable ASCII, both walk straight past it. It catches the
    careless case cheaply. What bounds the damage is that the review job's
    credential is Bedrock-only and expires in fifteen minutes.
    """
    for match in _BLOB.finditer(text):
        if _decodes_to_binary(match.group(0)):
            raise Rejected(
                f"{where}: contains a {len(match.group(0))}-char run that decodes to "
                f"non-printable bytes, which is indistinguishable from a smuggled payload"
            )


def check_value_blobs(obj, where: str) -> None:
    for text in iter_strings(obj):
        check_no_encoded_blob(text, where)


def neutralize(text: str, cap: int = MAX_SUMMARY) -> str:
    """Defuse rendering hazards without changing the reviewer's meaning.

    TRUNCATION HAPPENS HERE, FIRST, and that is a correctness property rather
    than tidiness. ``_MD_IMAGE`` and ``_MD_LINK`` backtrack from every start
    position, so this function is quadratic in its input. Capping the RESULT
    instead would leave the input unbounded, and unbounded costs 0.45s at 16KB,
    1.81s at 32KB, 6.78s at 64KB — a clean 4x per doubling, so
    ``{"summary": "![" * 131072}`` costs minutes and the job
    dies with no artifact at all, which the telemetry cannot tell apart from a
    dead runner. Capping the input first bounds the whole thing.

    OUT-OF-CHARSET CODEPOINTS ARE DROPPED HERE, not rejected downstream. The
    assembled result is charset-checked and a failure discards the WHOLE review,
    so leaving this to that check made one codepoint a complete denial of
    service — and not only under attack: a model writes an em dash or a curly
    apostrophe in ordinary prose, and that alone would have thrown away every
    finding in the review plus its telemetry row. Dropping the codepoint removes
    the same steganographic channel (zero-width joiners, RTL overrides) without
    handing anyone a one-character kill switch.

    ORDER MATTERS. Tags are stripped FIRST, because a tag that VANISHES can act
    as a separator: strip ``<i></i>`` last and ``https:<i></i>//evil``
    reassembles into a live URL after the URL rule here has already run.

    The ordering does NOT matter for the encoded-blob check, despite looking as
    though it should: ``check_value_blobs`` runs in ``build`` on the fully
    assembled post-neutralize result, so it sees every internal ordering. The
    ``_URL`` half is the whole of the argument.

    Backslashes are removed before the URL pass, because CommonMark honours
    escapes inside a link destination: ``\\/\\/evil.example.com`` renders as
    ``//evil.example.com`` while containing no ``//`` for ``_URL`` to match.  # @lint-ignore

    Residual ``<``, ``>`` and ``&`` are escaped LAST, which closes what
    stripping structurally cannot: an unterminated ``<!--`` has no closing
    ``>``, so no tag pattern can match it, and left alone it comments out
    everything a renderer shows after it.

    Defusing uses VISIBLE ASCII (``@ user``, ``\\# 1234``). Do not reach for a
    zero-width space: it reads better and is exactly the codepoint class this
    function strips. The backslash on the hash is not decoration — ``#123`` is
    NOT a markdown heading (ATX needs a space after the hashes) but a bare
    ``# 123`` IS one at line start, so defusing without it MANUFACTURES the
    hazard it is defusing.

    LINK SYNTAX IS CLOSED BY ESCAPING, NOT BY PEELING. No bare ``[`` or ``]``
    reaches the output. The peel loop is readability — it turns a real
    ``[text](dest)`` into ``text`` instead of into escaped punctuation — and it
    could never have been the control, because a regex cannot match a balanced
    bracket label: ``_MD_LINK``'s ``[^\\]]*`` stops at the first ``]``, so
    ``[[x]](/evil)`` is a live CommonMark link it never matched. Reference
    links, shortcut references, relative destinations and ``javascript:`` all
    reached the output the same way. The escape at the end takes the whole class
    at once, since every one of them needs a surviving bracket pair.

    Those backslashes are added AFTER the strip above and only ever before a
    bracket, so they cannot reconstitute the ``\\/\\/host`` case that strip
    exists for.

    NOT DONE HERE, and it is a real gap: markdown STRUCTURE the attacker writes
    is not escaped, so a summary can still forge a heading, a table, a
    horizontal rule or a fenced block. Nothing renders these strings today — the
    row carries no findings and Dr.CI rendering is unbuilt — so this is latent.
    It has to be closed in the same change that starts rendering, and the right
    escaping depends on that renderer.
    """
    text = "".join(c for c in text[:cap] if _ALLOWED_CHARS.match(c))
    text = _HTML_TAG.sub("", text.replace("\\", ""))
    # Peel links down to their text. This loop is READABILITY, not the control
    # — the escape below is the control — so its bound is allowed to be a
    # constant. `.sub()` is a single non-recursive pass and the replacement is
    # the link TEXT, which may itself contain `[`, so one pass peels exactly one
    # nesting level: `[[Sign in](x)](//evil)` becomes `[Sign in](//evil)`. @lint-ignore
    # Nesting costs four characters a level (`[]()`, then `[Dn]()`), so a
    # 1500-character summary funds depth 375 and no constant covers every case.
    # What is past the bound comes out escaped instead of tidy, which is a
    # cosmetic loss and not a hole. 64 is also where the cost lives: a summary
    # of many unmatched `[` before a distant `]` makes each pass quadratic, and
    # measured worst case is 0.9 ms for pure nesting but 627 ms for that shape.
    for _ in range(64):
        rewritten = _MD_LINK.sub(r"\1", _MD_IMAGE.sub(_IMAGE_MARKER, text))
        if rewritten == text:
            break
        text = rewritten
    # ESCAPE EVERY SURVIVING BRACKET, and this is the control. A regex cannot
    # match a balanced-bracket link label — `_MD_LINK`'s `[^\]]*` stops at the
    # first `]` — so peeling structurally cannot see the forms below, and no
    # bound on the loop above would have helped:
    #   * `[[x]](/evil)` — CommonMark allows a matched pair inside link text,
    #     so this is a live link that `_MD_LINK` never matches at all;
    #   * `[a][b]` with a `[b]: /evil` definition, and the shortcut `[a]` form;
    #   * a RELATIVE destination in any of them, which `_URL` does not defuse
    #     because it requires a scheme or `//host` — and `javascript:alert(1)`,
    #     which is a live destination `_URL` also misses (it wants `://`).
    # Every one of those needs a surviving `[`...`]` pair, and none survives
    # this. `\[x\]` renders as the literal characters, so prose keeps its
    # meaning; `neutralize_path` already makes the same trade for paths.
    text = text.replace("[", "\\[").replace("]", "\\]")
    text = _URL.sub(_LINK_MARKER, text)
    text = _XREF.sub(r"\1\\# \2", text)
    text = _MENTION.sub(r"@ \1", text)
    text = _ISSUE_REF.sub(r"\\# \1", text)
    for char, escaped in _HTML_ESCAPES:
        text = text.replace(char, escaped)
    # Capped on the way out as well as in: escaping GROWS the string, and one
    # place deciding the length beats every caller re-truncating.
    return text[:cap]


def unquote_git_path(path: str) -> str:
    """Decode a path the way ``git diff`` quotes it when it has special bytes.

    ``codecs.escape_decode`` is an undocumented CPython-private API with no clean
    stdlib equivalent. It has changed across minor releases before, so treat it
    as a dependency to re-check on interpreter upgrades. Failure is contained:
    both branches below fall back to the path minus its quotes, and the caller
    validates the result against the diff regardless, so a break here costs
    findings on exotic filenames rather than admitting an unvalidated path.
    """
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        try:
            return codecs.escape_decode(path[1:-1].encode())[0].decode(
                "utf-8", "replace"
            )
        except (ValueError, UnicodeDecodeError):
            return path[1:-1]
    return path


def neutralize_path(path: str) -> str:
    """Make a repo path safe to render WITHOUT changing which file it names.

    `neutralize` is wrong for a path: it rewrites rather than escapes, turning
    `node_modules/@babel/core/x.js` into `node_modules/@ babel/core/x.js` — not
    the file that was validated against the diff, not clickable, and not
    joinable with anything downstream.

    A path has already passed `_is_repo_path`, so the dangerous characters left
    are the markdown-active ones. Backslash-escaping each keeps the name exactly
    recoverable: `@babel` renders as `@babel` and notifies nobody, `x[0].py`
    renders as `x[0].py` and links nowhere, `#1.py` renders as `#1.py` and
    cross-references nothing.
    """
    return "".join("\\" + c if c in "[]()@#*_" else c for c in path)


def _is_repo_path(path: str) -> bool:
    """Reject anything that is not a plausible repo-relative path.

    The diff is attacker-authored, and its ``+++`` targets become the published
    ``path`` of a finding. Without this, ``+++ /etc/shadow`` published
    ``/etc/shadow`` and ``+++ b/../../x`` published ``../../x`` — a finding that
    points outside the repository, in a surface a reader takes as the tool's own
    statement about the change. Nothing here reads the path, so this is about
    what may be SAID, not about traversal.
    """
    if not path or len(path) > MAX_PATH:
        return False
    # A leading `/` reads as absolute and a leading `~` reads as a home path;
    # neither can be a repo-relative name.
    if path[0] in "/~":
        return False
    # `|` and backtick are refused for the same reason tab is: a path renders as
    # a cell or inline code, and both are structural there. Refusing tab while
    # allowing the pipe would have closed the awkward spelling of table-row
    # forgery and left the natural one open. `\` is refused because it reads as
    # a UNC path (`\\server\share`) and because backslash escapes are a renderer
    # trick — `neutralize` strips them from prose for the same reason.
    if any(c in path for c in "|`\\"):
        return False
    # Printable ASCII only, and NOT tab — `_ALLOWED_CHARS` permits tab, but a
    # path is rendered as a cell rather than as prose, and a tab is a column
    # separator. `git diff` quotes such a name, and `unquote_git_path` faithfully
    # decodes it, so `"b/evil\t| fake | row |"` reached the output verbatim.
    #
    # Refusing the anchor HERE also keeps an availability bug narrow. A file
    # with a non-ASCII name is perfectly legal; letting it through would fail
    # the charset check on the assembled result, which rejects the ENTIRE
    # review rather than the one finding. Dropping it here costs that finding
    # alone and leaves the rest of the review publishable.
    if not _PATH_CHARS.match(path):
        return False
    return not any(part in ("", ".", "..") for part in path.split("/"))


def parse_diff(diff_text: str) -> dict[str, set[int]]:
    """Map each changed file to the set of new-side line numbers the diff touches.

    Only the post-image (``+++ b/...``) side is recorded, because that is what a
    finding's ``line`` refers to. Both added and context lines inside a hunk
    count, so a reviewer may anchor a finding on an unchanged line next to the
    change.

    HEADERS ARE ONLY RECOGNISED OUTSIDE A HUNK BODY, and the hunk's own declared
    line counts are what say where the body ends. This is the anchoring
    guarantee, and the obvious weaker rule does not provide it.

    The weaker rule — "honour ``+++`` only when it directly follows ``---``" —
    reasons that an ADDED source line reading ``++ fake.py`` renders as
    ``+++ fake.py`` with no ``---`` before it. True for an addition, and false
    for a MODIFICATION: change one line whose old text starts ``-- `` into new
    text starting ``++ `` and real ``git diff`` emits an adjacent
    ``--- ``/``+++ `` pair inside the hunk body. The deletion line satisfies the
    pairing, the pair is read as a file header, and since the line counter is
    not reset the forged file inherits whatever line numbers the last real
    ``@@`` set. That is an arbitrary path at arbitrary lines, which is exactly
    what "every finding names a file this PR changed" is supposed to prevent.

    ``@@ -a,b +c,d @@`` states exactly how many lines belong to the hunk, so the
    body is consumed by budget: context spends one from each side, ``+`` one
    from the new side, ``-`` one from the old. While either budget is open every
    line is body, whatever it looks like.

    KNOWN LIMITS, deliberate: pure renames, mode-only changes, binary files and
    deletions have no new-side hunk lines, so findings about them can never
    anchor and are dropped. Combined (merge) diffs are not parsed; a PR diff
    against its base is never one. A malformed diff whose body is shorter than
    its header declares leaves the budget open and swallows what follows — that
    loses anchors, never invents them, which is the safe direction.
    """
    touched: dict[str, set[int]] = {}
    current: str | None = None
    new_line = 0
    old_remaining = 0
    new_remaining = 0
    saw_git_header = False
    prev_was_old_header = False

    # `split("\n")`, NEVER `splitlines()`. Python splits on U+2028, U+2029,
    # \x0b, \x0c and \x85 as well as \n; git counts only \n. That disagreement
    # is a forgery primitive once line COUNTS are load-bearing: put one U+2028
    # inside a context line and git's `@@ -1,2 +1,2 @@` is honest while Python
    # sees an extra line, the budget closes early, and whatever the attacker put
    # after the separator is read as a file header outside any hunk.
    lines = diff_text.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]

        if old_remaining > 0 or new_remaining > 0:
            # Inside a hunk body. Every line here starts with ' ', '+', '-' or
            # '\' in real git output.
            if raw.startswith("\\"):
                i += 1
                continue  # "\ No newline at end of file" — spends no budget
            if raw.startswith("+"):
                new_remaining -= 1
            elif raw.startswith("-"):
                old_remaining -= 1
                i += 1
                continue  # deletions consume no new-side number
            elif raw == "" or raw.startswith(" "):
                # Context. A bare empty line is one too: some pipelines strip
                # the trailing space from an empty context line.
                old_remaining -= 1
                new_remaining -= 1
            else:
                # DESYNC. An unprefixed line cannot occur inside a real hunk —
                # a content line reading `diff --git ...` is emitted as
                # ` diff --git ...` — so the budget is wrong and continuing to
                # spend it would attribute a following stanza's lines to THIS
                # file and lose that stanza entirely. Close the hunk and
                # reprocess this same line as a header, without advancing.
                old_remaining = new_remaining = 0
                continue
            if current is not None:
                touched[current].add(new_line)
            new_line += 1
            i += 1
            continue

        i += 1
        if raw.startswith("diff --git "):
            # A new stanza always closes the previous file, even if its hunks
            # were malformed.
            current = None
            saw_git_header = True
            prev_was_old_header = False
            continue
        if raw.startswith("--- "):
            prev_was_old_header = True
            current = None
            continue
        if raw.startswith("+++ ") and prev_was_old_header and saw_git_header:
            # `saw_git_header` is the third guard. Real `git diff` always emits
            # `diff --git a/X b/X` before the `---`/`+++` pair, so a pair that
            # appears without one did not come from git and must not name a
            # file. Without it, any way of ending a hunk early hands the
            # attacker a header position.
            prev_was_old_header = False
            saw_git_header = False
            target = unquote_git_path(raw[4:].strip())
            if target == "/dev/null":
                current = None
            else:
                target = target[2:] if target.startswith(("a/", "b/")) else target
                current = target if _is_repo_path(target) else None
            if current is not None:
                touched.setdefault(current, set())
            continue
        prev_was_old_header = False

        if raw.startswith("@@"):
            m = re.match(r"@@+ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", raw)
            if m and current is not None:
                old_remaining = int(m.group(2)) if m.group(2) is not None else 1
                new_line = int(m.group(3))
                new_remaining = int(m.group(4)) if m.group(4) is not None else 1
            else:
                # Unparsable, or a hunk with no file to attach it to. Record
                # nothing rather than guess.
                old_remaining = new_remaining = 0
                new_line = 0
                current = None
            continue
    return touched


def sanitize_findings(
    raw_findings, touched: dict[str, set[int]]
) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    dropped: list[dict] = []

    def drop(record: dict) -> None:
        # Bounded: an unbounded findings array would otherwise let the model
        # grow this list without limit. Every attacker-derived string in a
        # record is neutralized and capped, because `dropped_detail` is
        # published and would otherwise be an uncapped side channel that
        # bypasses the caps applied to real findings.
        if len(dropped) < MAX_DROPPED_TRACKED:
            if "path" in record and isinstance(record["path"], str):
                record["path"] = neutralize(record["path"], MAX_PATH)[:MAX_PATH]
            if "keys" in record:
                record["keys"] = [neutralize(str(k), 64)[:64] for k in record["keys"]]
            dropped.append(record)

    for item in raw_findings or []:
        if not isinstance(item, dict):
            drop({"reason": "not_an_object"})
            continue
        if len(kept) >= MAX_FINDINGS:
            drop({"reason": "over_cap"})
            continue

        # Re-validate the types the producer's JSON schema was supposed to
        # guarantee. The producer is attacker-influenced, so "the schema already
        # checked it" is not a property this boundary may assume — and a
        # permissive str()/int() here would launder a wrong-typed value into a
        # published verdict.
        extra_keys = set(item) - {"path", "line", "severity", "message"}
        if extra_keys:
            drop({"reason": "unexpected_keys", "keys": sorted(extra_keys)[:5]})
            continue
        path = item.get("path")
        message = item.get("message")
        if not isinstance(path, str) or not isinstance(message, str):
            drop({"reason": "path_or_message_not_a_string"})
            continue
        path, message = path.strip(), message.strip()
        if not path or not message:
            drop({"path": path, "reason": "missing_path_or_message"})
            continue
        if path not in touched:
            drop({"path": path, "reason": "path_not_in_diff"})
            continue
        line = item.get("line")
        # bool is a subclass of int; True must not silently become line 1.
        if not isinstance(line, int) or isinstance(line, bool):
            drop({"path": path, "reason": "line_not_an_integer"})
            continue
        if line not in touched[path]:
            drop({"path": path, "line": line, "reason": "line_not_in_diff"})
            continue
        severity = item.get("severity")
        if not isinstance(severity, str) or severity.lower() not in SEVERITIES:
            drop({"path": path, "reason": "bad_severity"})
            continue

        clean_message = neutralize(message, MAX_MESSAGE).strip()[:MAX_MESSAGE].strip()
        if not clean_message:
            # e.g. a message consisting only of `<b></b>`: it survives the
            # non-empty check above but neutralizes to nothing, and an empty
            # finding is not evidence of anything.
            drop({"path": path, "reason": "message_empty_after_neutralize"})
            continue

        kept.append(
            {
                # Neutralized like every other published string. It was the one
                # that was not — `drop()` neutralized its copy while the KEPT
                # copy was merely truncated, and the kept one is the copy that
                # gets rendered. A path is attacker-chosen text: `@handle`,
                # `[label](target)` and `#1234` are all legal in a filename, so
                # `touch '@pytorchbot [rebase](//evil) #1.py'` @lint-ignore
                # was enough to put a live mention and a live link into a
                # bot-authored surface, with no diff trickery at all.
                #
                # `neutralize_path`, not `neutralize`: this one has to stay the
                # NAME of the file it was validated against.
                "path": neutralize_path(path)[:MAX_PATH],
                "line": line,
                "severity": severity.lower(),
                "message": clean_message,
            }
        )
    return kept, dropped


def build(obj: dict, touched: dict[str, set[int]]) -> dict:
    # NO charset check on the INPUT, deliberately. One there would make a single
    # out-of-charset codepoint anywhere in the model's output discard the entire
    # review — an em dash or a curly apostrophe in ordinary review prose, which
    # a model writes without any prompting, and a guaranteed one-character
    # denial of service under injection. Every string that reaches the artifact
    # goes through `neutralize`, which DROPS those codepoints instead; the
    # `check_value_charset` on the assembled result below is the assertion that
    # this actually held.
    #
    # The blob check stays on the input. It is not a formatting rule — a long
    # encoding-shaped run that decodes to binary is a smuggling signal, and the
    # right response to a signal is to refuse the review, not to tidy it away.
    check_value_blobs(obj, "model output")

    extra_keys = set(obj) - {"verdict", "summary", "findings"}
    if extra_keys:
        raise Rejected(f"unexpected top-level key(s): {sorted(extra_keys)[:5]}")

    verdict = obj.get("verdict")
    if not isinstance(verdict, str) or verdict.strip() not in VERDICTS:
        raise Rejected(f"verdict is not one of {sorted(VERDICTS)}")
    verdict = verdict.strip()

    summary_raw = obj.get("summary")
    if not isinstance(summary_raw, str):
        raise Rejected("summary is not a string")
    summary = neutralize(summary_raw, MAX_SUMMARY).strip()[:MAX_SUMMARY].strip()
    if not summary:
        raise Rejected("summary is empty after neutralization")

    findings_raw = obj.get("findings")
    if not isinstance(findings_raw, list):
        raise Rejected("findings is not an array")

    kept, dropped = sanitize_findings(findings_raw, touched)

    # An objection must arrive with evidence. `changes_requested` and no
    # surviving finding is refused whether the findings were discarded as
    # unanchored or never supplied: either way a reader would be told the change
    # needs work with nothing to point at, and the likeliest route here is a
    # model talked into objecting to a file the PR never touched.
    if verdict == "changes_requested" and not kept:
        raise Rejected(
            f"verdict is changes_requested with no publishable finding "
            f"({len(dropped)} discarded) — refusing to publish an objection with no evidence"
        )

    result = {
        "status": "succeeded",
        "verdict": verdict,
        "summary": summary,
        "findings": kept,
        "findings_dropped": len(dropped),
        "dropped_detail": dropped[:MAX_FINDINGS],
    }
    # Validate what we are about to WRITE, not just what we read. neutralize()
    # and the caps run between those two points; a cap can also truncate a
    # string mid-sequence, and neutralize() could in principle reassemble a
    # blob. Re-checking here is what makes the module's advertised invariants
    # true of the artifact rather than of the input.
    check_value_charset(result, "sanitized result")
    check_value_blobs(result, "sanitized result")
    return result


def failure(status: str, reason: str) -> dict:
    # `reason` can quote attacker-influenced text, so it gets the same
    # neutralization and cap as any other published string.
    detail = neutralize(str(reason), MAX_SUMMARY)[:MAX_SUMMARY]
    detail = "".join(c for c in detail if _ALLOWED_CHARS.match(c))
    return {
        "status": status,
        "verdict": None,
        "summary": "",
        "findings": [],
        "findings_dropped": 0,
        "dropped_detail": [],
        "failure_detail": detail,
    }


def downgrade(result: dict, outcome: str) -> dict:
    """Re-status a result whose producing step did not succeed.

    A structured output can be perfectly well-formed while the step that emitted
    it failed, timed out, was cancelled, or never ran because an earlier step
    failed closed. Publishing `succeeded` for any of those would tell a reader
    the review happened.

    `cancelled` and `skipped` both mean the review was PREVENTED — a superseding
    push, or an upstream step that fail-closed (an unreachable base commit, say).
    That is `blocked`. `failure` means the model step itself ran and broke, which
    is `model_error`. Collapsing the two loses the only signal that distinguishes
    our own infrastructure faulting from the model faulting.
    """
    status = "blocked" if outcome in ("cancelled", "skipped") else "model_error"
    detail = neutralize(str(outcome), MAX_SUMMARY)[:MAX_SUMMARY]
    detail = "".join(c for c in detail if _ALLOWED_CHARS.match(c))
    return {
        **result,
        "status": status,
        "verdict": None,
        "failure_detail": f"claude step outcome={detail}",
    }


def load_structured(path: Path) -> tuple[dict | None, str]:
    """Return (parsed_object, failure_status). Never raises."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"model_error:could not read structured output: {exc}"
    # A ceiling on the whole document, before anything walks it. The per-field
    # caps bound what gets PUBLISHED; they do not bound what gets parsed and
    # scanned, and every check in this module is at least linear in the input.
    # 1 MiB is ~700x the publishable summary, so nothing legitimate is near it.
    if len(raw) > MAX_STRUCTURED_BYTES:
        return None, (
            f"schema_invalid:structured output is {len(raw)}B, "
            f"over the {MAX_STRUCTURED_BYTES}B cap"
        )
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return None, "model_error:claude-code-action produced no structured_output"
    try:
        parsed = json.loads(text)
    except RecursionError:
        # Deeply nested JSON is a cheap way to crash the boundary; a crash means
        # no artifact, which the telemetry cannot distinguish from a dead runner.
        return None, "schema_invalid:structured output nests too deeply to parse"
    except ValueError as exc:
        return None, f"schema_invalid:{exc}"
    if not isinstance(parsed, dict):
        return None, "schema_invalid:top-level structured output is not an object"
    return parsed, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured-output-file", required=True)
    ap.add_argument("--diff-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--step-outcome",
        default="success",
        help="GitHub outcome of the step that produced the structured output; "
        "anything but 'success' downgrades the published status",
    )
    args = ap.parse_args()

    try:
        obj, problem = load_structured(Path(args.structured_output_file))
        if obj is None:
            status, _, detail = problem.partition(":")
            result = failure(status, detail)
        else:
            try:
                diff_text = Path(args.diff_file).read_bytes().decode("utf-8", "replace")
            except OSError as exc:
                # Do NOT fall back to an empty diff. With no diff, anchoring
                # never runs, every finding is discarded, and a
                # `ready_for_human_review` verdict would be published as if the
                # review had actually happened.
                result = failure("blocked", f"could not read the diff: {exc}")
            else:
                try:
                    result = build(obj, parse_diff(diff_text))
                except Rejected as exc:
                    result = failure("sanitizer_rejected", str(exc))
                except RecursionError:
                    result = failure(
                        "schema_invalid", "structured output nests too deeply"
                    )
                except (ValueError, TypeError) as exc:
                    result = failure("schema_invalid", str(exc))
    except Exception as exc:  # noqa: BLE001 - the boundary must always emit a row
        result = failure(
            "model_error", f"unexpected sanitizer error: {type(exc).__name__}"
        )

    # Applied last, and outside the try: a non-success outcome overrides
    # whatever the content said, including a clean `succeeded`.
    if args.step_outcome != "success":
        result = downgrade(result, args.step_outcome)

    try:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=True))
    except OSError as exc:
        print(f"FATAL: could not write {args.out}: {exc}", file=sys.stderr)
        return 1

    print(
        f"status={result['status']} verdict={result.get('verdict')} "
        f"findings={len(result['findings'])} dropped={result['findings_dropped']}",
        file=sys.stderr,
    )
    if result.get("failure_detail"):
        print(f"detail: {result['failure_detail']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import logging
from os.path import basename, splitext
from typing import BinaryIO, List, Optional, Set

try:
    from os import PathLike
except ImportError:  # pragma: no cover
    PathLike = str  # type: ignore

from .cd import (
    coherence_ratio,
    encoding_languages,
    mb_encoding_languages,
    merge_coherence_ratios,
)
from .constant import IANA_SUPPORTED, TOO_BIG_SEQUENCE, TOO_SMALL_SEQUENCE, TRACE
from .md import mess_ratio
from .models import CharsetMatch, CharsetMatches
from .utils import (
    any_specified_encoding,
    iana_name,
    identify_sig_or_bom,
    is_cp_similar,
    is_multi_byte_encoding,
    should_strip_sig_or_bom,
)

# Will most likely be controversial
# logging.addLevelName(TRACE, "TRACE")
logger = logging.getLogger("charset_normalizer")
explain_handler = logging.StreamHandler()
explain_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)


def from_bytes(
    sequences: bytes,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.2,
    cp_isolation: List[str] = None,
    cp_exclusion: List[str] = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
) -> CharsetMatches:
    """
    Given a raw bytes sequence, return the best possibles charset usable to render str objects.
    If there is no results, it is a strong indicator that the source is binary/not text.
    By default, the process will extract 5 blocs of 512o each to assess the mess and coherence of a given sequence.
    And will give up a particular code page after 20% of measured mess. Those criteria are customizable at will.

    The preemptive behavior DOES NOT replace the traditional detection workflow, it prioritize a particular code page
    but never take it for granted. Can improve the performance.

    You may want to focus your attention to some code page or/and not others, use cp_isolation and cp_exclusion for that
    purpose.

    This function will strip the SIG in the payload/sequence every time except on UTF-16, UTF-32.
    By default the library does not setup any handler other than the NullHandler, if you choose to set the 'explain'
    toggle to True it will alter the logger configuration to add a StreamHandler that is suitable for debugging.
    Custom logging format and handler can be set manually.
    """

    if not isinstance(sequences, (bytearray, bytes)):
        raise TypeError(
            "Expected object of type bytes or bytearray, got: {0}".format(
                type(sequences)
            )
        )

    if explain:
        previous_logger_level = logger.level  # type: int
        logger.addHandler(explain_handler)
        logger.setLevel(TRACE)

    length = len(sequences)  # type: int

    if length == 0:
        logger.debug("Encoding detection on empty bytes, assuming utf_8 intention.")
        if explain:
            logger.removeHandler(explain_handler)
            logger.setLevel(previous_logger_level or logging.WARNING)
        return CharsetMatches([CharsetMatch(sequences, "utf_8", 0.0, False, [], "")])

    if cp_isolation is not None:
        logger.log(
            TRACE,
            "cp_isolation is set. use this flag for debugging purpose. "
            "limited list of encoding allowed : %s.",
            ", ".join(cp_isolation),
        )
        cp_isolation = [iana_name(cp, False) for cp in cp_isolation]
    else:
        cp_isolation = []

    if cp_exclusion is not None:
        logger.log(
            TRACE,
            "cp_exclusion is set. use this flag for debugging purpose. "
            "limited list of encoding excluded : %s.",
            ", ".join(cp_exclusion),
        )
        cp_exclusion = [iana_name(cp, False) for cp in cp_exclusion]
    else:
        cp_exclusion = []

    if length <= (chunk_size * steps):
        logger.log(
            TRACE,
            "override steps (%i) and chunk_size (%i) as content does not fit (%i byte(s) given) parameters.",
            steps,
            chunk_size,
            length,
        )
        steps = 1
        chunk_size = length

    if steps > 1 and length / steps < chunk_size:
        chunk_size = int(length / steps)

    is_too_small_sequence = len(sequences) < TOO_SMALL_SEQUENCE  # type: bool
    is_too_large_sequence = len(sequences) >= TOO_BIG_SEQUENCE  # type: bool

    if is_too_small_sequence:
        logger.log(
            TRACE,
            "Trying to detect encoding from a tiny portion of ({}) byte(s).".format(
                length
            ),
        )
    elif is_too_large_sequence:
        logger.log(
            TRACE,
            "Using lazy str decoding because the payload is quite large, ({}) byte(s).".format(
                length
            ),
        )

    prioritized_encodings = []  # type: List[str]

    specified_encoding = (
        any_specified_encoding(sequences) if preemptive_behaviour else None
    )  # type: Optional[str]

    if specified_encoding is not None:
        prioritized_encodings.append(specified_encoding)
        logger.log(
            TRACE,
            "Detected declarative mark in sequence. Priority +1 given for %s.",
            specified_encoding,
        )

    tested = set()  # type: Set[str]
    tested_but_hard_failure = []  # type: List[str]
    tested_but_soft_failure = []  # type: List[str]

    fallback_ascii = None  # type: Optional[CharsetMatch]
    fallback_u8 = None  # type: Optional[CharsetMatch]
    fallback_specified = None  # type: Optional[CharsetMatch]

    results = CharsetMatches()  # type: CharsetMatches

    sig_encoding, sig_payload = identify_sig_or_bom(sequences)

    if sig_encoding is not None:
        prioritized_encodings.append(sig_encoding)
        logger.log(
            TRACE,
            "Detected a SIG or BOM mark on first %i byte(s). Priority +1 given for %s.",
            len(sig_payload),
            sig_encoding,
        )

    prioritized_encodings.append("ascii")

    if "utf_8" not in prioritized_encodings:
        prioritized_encodings.append("utf_8")

    for encoding_iana in prioritized_encodings + IANA_SUPPORTED:

        if cp_isolation and encoding_iana not in cp_isolation:
            continue

        if cp_exclusion and encoding_iana in cp_exclusion:
            continue

        if encoding_iana in tested:
            continue

        tested.add(encoding_iana)

        decoded_payload = None  # type: Optional[str]
        bom_or_sig_available = sig_encoding == encoding_iana  # type: bool
        strip_sig_or_bom = bom_or_sig_available and should_strip_sig_or_bom(
            encoding_iana
        )  # type: bool

        if encoding_iana in {"utf_16", "utf_32"} and not bom_or_sig_available:
            logger.log(
                TRACE,
                "Encoding %s wont be tested as-is because it require a BOM. Will try some sub-encoder LE/BE.",
                encoding_iana,
            )
            continue

        try:
            is_multi_byte_decoder = is_multi_byte_encoding(encoding_iana)  # type: bool
        except (ModuleNotFoundError, ImportError):
            logger.log(
                TRACE,
                "Encoding %s does not provide an IncrementalDecoder",
                encoding_iana,
            )
            continue

        try:
            if is_too_large_sequence and is_multi_byte_decoder is False:
                str(
                    sequences[: int(50e4)]
                    if strip_sig_or_bom is False
                    else sequences[len(sig_payload) : int(50e4)],
                    encoding=encoding_iana,
                )
            else:
                decoded_payload = str(
                    sequences
                    if strip_sig_or_bom is False
                    else sequences[len(sig_payload) :],
                    encoding=encoding_iana,
                )
        except (UnicodeDecodeError, LookupError) as e:
            if not isinstance(e, LookupError):
                logger.log(
                    TRACE,
                    "Code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
            tested_but_hard_failure.append(encoding_iana)
            continue

        similar_soft_failure_test = False  # type: bool

        for encoding_soft_failed in tested_but_soft_failure:
            if is_cp_similar(encoding_iana, encoding_soft_failed):
                similar_soft_failure_test = True
                break

        if similar_soft_failure_test:
            logger.log(
                TRACE,
                "%s is deemed too similar to code page %s and was consider unsuited already. Continuing!",
                encoding_iana,
                encoding_soft_failed,
            )
            continue

        r_ = range(
            0 if not bom_or_sig_available else len(sig_payload),
            length,
            int(length / steps),
        )

        multi_byte_bonus = (
            is_multi_byte_decoder
            and decoded_payload is not None
            and len(decoded_payload) < length
        )  # type: bool

        if multi_byte_bonus:
            logger.log(
                TRACE,
                "Code page %s is a multi byte encoding table and it appear that at least one character "
                "was encoded using n-bytes.",
                encoding_iana,
            )

        max_chunk_gave_up = int(len(r_) / 4)  # type: int

        max_chunk_gave_up = max(max_chunk_gave_up, 2)
        early_stop_count = 0  # type: int
        lazy_str_hard_failure = False

        md_chunks = []  # type: List[str]
        md_ratios = []

        for i in r_:
            if i + chunk_size > length + 8:
                continue

            cut_sequence = sequences[i : i + chunk_size]

            if bom_or_sig_available and strip_sig_or_bom is False:
                cut_sequence = sig_payload + cut_sequence

            try:
                chunk = cut_sequence.decode(
                    encoding_iana,
                    errors="ignore" if is_multi_byte_decoder else "strict",
                )  # type: str
            except UnicodeDecodeError as e:  # Lazy str loading may have missed something there
                logger.log(
                    TRACE,
                    "LazyStr Loading: After MD chunk decode, code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
                early_stop_count = max_chunk_gave_up
                lazy_str_hard_failure = True
                break

            # multi-byte bad cutting detector and adjustment
            # not the cleanest way to perform that fix but clever enough for now.
            if is_multi_byte_decoder and i > 0 and sequences[i] >= 0x80:

                chunk_partial_size_chk = min(chunk_size, 16)  # type: int

                if (
                    decoded_payload
                    and chunk[:chunk_partial_size_chk] not in decoded_payload
                ):
                    for j in range(i, i - 4, -1):
                        cut_sequence = sequences[j : i + chunk_size]

                        if bom_or_sig_available and strip_sig_or_bom is False:
                            cut_sequence = sig_payload + cut_sequence

                        chunk = cut_sequence.decode(encoding_iana, errors="ignore")

                        if chunk[:chunk_partial_size_chk] in decoded_payload:
                            break

            md_chunks.append(chunk)

            md_ratios.append(mess_ratio(chunk, threshold))

            if md_ratios[-1] >= threshold:
                early_stop_count += 1

            if (early_stop_count >= max_chunk_gave_up) or (
                bom_or_sig_available and strip_sig_or_bom is False
            ):
                break

        # We might want to check the sequence again with the whole content
        # Only if initial MD tests passes
        if (
            not lazy_str_hard_failure
            and is_too_large_sequence
            and not is_multi_byte_decoder
        ):
            try:
                sequences[int(50e3) :].decode(encoding_iana, errors="strict")
            except UnicodeDecodeError as e:
                logger.log(
                    TRACE,
                    "LazyStr Loading: After final lookup, code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
                tested_but_hard_failure.append(encoding_iana)
                continue

        mean_mess_ratio = (
            sum(md_ratios) / len(md_ratios) if md_ratios else 0.0
        )  # type: float
        if mean_mess_ratio >= threshold or early_stop_count >= max_chunk_gave_up:
            tested_but_soft_failure.append(encoding_iana)
            logger.log(
                TRACE,
                "%s was excluded because of initial chaos probing. Gave up %i time(s). "
                "Computed mean chaos is %f %%.",
                encoding_iana,
                early_stop_count,
                round(mean_mess_ratio * 100, ndigits=3),
            )
            # Preparing those fallbacks in case we got nothing.
            if (
                encoding_iana in ["ascii", "utf_8", specified_encoding]
                and not lazy_str_hard_failure
            ):
                fallback_entry = CharsetMatch(
                    sequences, encoding_iana, threshold, False, [], decoded_payload
                )
                if encoding_iana == specified_encoding:
                    fallback_specified = fallback_entry
                elif encoding_iana == "ascii":
                    fallback_ascii = fallback_entry
                else:
                    fallback_u8 = fallback_entry
            continue

        logger.log(
            TRACE,
            "%s passed initial chaos probing. Mean measured chaos is %f %%",
            encoding_iana,
            round(mean_mess_ratio * 100, ndigits=3),
        )

        if not is_multi_byte_decoder:
            target_languages = encoding_languages(encoding_iana)  # type: List[str]
        else:
            target_languages = mb_encoding_languages(encoding_iana)

        if target_languages:
            logger.log(
                TRACE,
                "{} should target any language(s) of {}".format(
                    encoding_iana, str(target_languages)
                ),
            )

        cd_ratios = []

        # We shall skip the CD when its about ASCII
        # Most of the time its not relevant to run "language-detection" on it.
        if encoding_iana != "ascii":
            for chunk in md_chunks:
                chunk_languages = coherence_ratio(
                    chunk, 0.1, ",".join(target_languages) if target_languages else None
                )

                cd_ratios.append(chunk_languages)

        cd_ratios_merged = merge_coherence_ratios(cd_ratios)

        if cd_ratios_merged:
            logger.log(
                TRACE,
                "We detected language {} using {}".format(
                    cd_ratios_merged, encoding_iana
                ),
            )

        results.append(
            CharsetMatch(
                sequences,
                encoding_iana,
                mean_mess_ratio,
                bom_or_sig_available,
                cd_ratios_merged,
                decoded_payload,
            )
        )

        if (
            encoding_iana in [specified_encoding, "ascii", "utf_8"]
            and mean_mess_ratio < 0.1
        ):
            logger.debug(
                "Encoding detection: %s is most likely the one.", encoding_iana
            )
            if explain:
                logger.removeHandler(explain_handler)
                logger.setLevel(previous_logger_level)
            return CharsetMatches([results[encoding_iana]])

        if encoding_iana == sig_encoding:
            logger.debug(
                "Encoding detection: %s is most likely the one as we detected a BOM or SIG within "
                "the beginning of the sequence.",
                encoding_iana,
            )
            if explain:
                logger.removeHandler(explain_handler)
                logger.setLevel(previous_logger_level)
            return CharsetMatches([results[encoding_iana]])

    if len(results) == 0:
        if fallback_u8 or fallback_ascii or fallback_specified:
            logger.log(
                TRACE,
                "Nothing got out of the detection process. Using ASCII/UTF-8/Specified fallback.",
            )

        if fallback_specified:
            logger.debug(
                "Encoding detection: %s will be used as a fallback match",
                fallback_specified.encoding,
            )
            results.append(fallback_specified)
        elif (
            (fallback_u8 and fallback_ascii is None)
            or (
                fallback_u8
                and fallback_ascii
                and fallback_u8.fingerprint != fallback_ascii.fingerprint
            )
            or (fallback_u8 is not None)
        ):
            logger.debug("Encoding detection: utf_8 will be used as a fallback match")
            results.append(fallback_u8)
        elif fallback_ascii:
            logger.debug("Encoding detection: ascii will be used as a fallback match")
            results.append(fallback_ascii)

    if results:
        logger.debug(
            "Encoding detection: Found %s as plausible (best-candidate) for content. With %i alternatives.",
            results.best().encoding,  # type: ignore
            len(results) - 1,
        )
    else:
        logger.debug("Encoding detection: Unable to determine any suitable charset.")

    if explain:
        logger.removeHandler(explain_handler)
        logger.setLevel(previous_logger_level)

    return results


def from_fp(
    fp: BinaryIO,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: List[str] = None,
    cp_exclusion: List[str] = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
) -> CharsetMatches:
    """
    Same thing than the function from_bytes but using a file pointer that is already ready.
    Will not close the file pointer.
    """
    return from_bytes(
        fp.read(),
        steps,
        chunk_size,
        threshold,
        cp_isolation,
        cp_exclusion,
        preemptive_behaviour,
        explain,
    )


def from_path(
    path: PathLike,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: List[str] = None,
    cp_exclusion: List[str] = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
) -> CharsetMatches:
    """
    Same thing than the function from_bytes but with one extra step. Opening and reading given file path in binary mode.
    Can raise IOError.
    """
    with open(path, "rb") as fp:
        return from_fp(
            fp,
            steps,
            chunk_size,
            threshold,
            cp_isolation,
            cp_exclusion,
            preemptive_behaviour,
            explain,
        )


def normalize(
    path: PathLike,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: List[str] = None,
    cp_exclusion: List[str] = None,
    preemptive_behaviour: bool = True,
) -> CharsetMatch:
    """
    Take a (text-based) file path and try to create another file next to it, this time using UTF-8.
    """
    results = from_path(
        path,
        steps,
        chunk_size,
        threshold,
        cp_isolation,
        cp_exclusion,
        preemptive_behaviour,
    )

    filename = basename(path)
    target_extensions = list(splitext(filename))

    if len(results) == 0:
        raise IOError(
            'Unable to normalize "{}", no encoding charset seems to fit.'.format(
                filename
            )
        )

    result = results.best()

    target_extensions[0] += "-" + result.encoding  # type: ignore

    with open(
        "{}".format(str(path).replace(filename, "".join(target_extensions))), "wb"
    ) as fp:
        fp.write(result.output())  # type: ignore

    return result  # type: ignore

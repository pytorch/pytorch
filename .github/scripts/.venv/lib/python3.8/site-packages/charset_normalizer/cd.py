import importlib
from codecs import IncrementalDecoder
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .assets import FREQUENCIES
from .constant import KO_NAMES, LANGUAGE_SUPPORTED_COUNT, TOO_SMALL_SEQUENCE, ZH_NAMES
from .md import is_suspiciously_successive_range
from .models import CoherenceMatches
from .utils import (
    is_accentuated,
    is_latin,
    is_multi_byte_encoding,
    is_unicode_range_secondary,
    unicode_range,
)


def encoding_unicode_range(iana_name: str) -> List[str]:
    """
    Return associated unicode ranges in a single byte code page.
    """
    if is_multi_byte_encoding(iana_name):
        raise IOError("Function not supported on multi-byte code page")

    decoder = importlib.import_module("encodings.{}".format(iana_name)).IncrementalDecoder  # type: ignore

    p = decoder(errors="ignore")  # type: IncrementalDecoder
    seen_ranges = {}  # type: Dict[str, int]
    character_count = 0  # type: int

    for i in range(0x40, 0xFF):
        chunk = p.decode(bytes([i]))  # type: str

        if chunk:
            character_range = unicode_range(chunk)  # type: Optional[str]

            if character_range is None:
                continue

            if is_unicode_range_secondary(character_range) is False:
                if character_range not in seen_ranges:
                    seen_ranges[character_range] = 0
                seen_ranges[character_range] += 1
                character_count += 1

    return sorted(
        [
            character_range
            for character_range in seen_ranges
            if seen_ranges[character_range] / character_count >= 0.15
        ]
    )


def unicode_range_languages(primary_range: str) -> List[str]:
    """
    Return inferred languages used with a unicode range.
    """
    languages = []  # type: List[str]

    for language, characters in FREQUENCIES.items():
        for character in characters:
            if unicode_range(character) == primary_range:
                languages.append(language)
                break

    return languages


@lru_cache()
def encoding_languages(iana_name: str) -> List[str]:
    """
    Single-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    unicode_ranges = encoding_unicode_range(iana_name)  # type: List[str]
    primary_range = None  # type: Optional[str]

    for specified_range in unicode_ranges:
        if "Latin" not in specified_range:
            primary_range = specified_range
            break

    if primary_range is None:
        return ["Latin Based"]

    return unicode_range_languages(primary_range)


@lru_cache()
def mb_encoding_languages(iana_name: str) -> List[str]:
    """
    Multi-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    if (
        iana_name.startswith("shift_")
        or iana_name.startswith("iso2022_jp")
        or iana_name.startswith("euc_j")
        or iana_name == "cp932"
    ):
        return ["Japanese"]
    if iana_name.startswith("gb") or iana_name in ZH_NAMES:
        return ["Chinese", "Classical Chinese"]
    if iana_name.startswith("iso2022_kr") or iana_name in KO_NAMES:
        return ["Korean"]

    return []


@lru_cache(maxsize=LANGUAGE_SUPPORTED_COUNT)
def get_target_features(language: str) -> Tuple[bool, bool]:
    """
    Determine main aspects from a supported language if it contains accents and if is pure Latin.
    """
    target_have_accents = False  # type: bool
    target_pure_latin = True  # type: bool

    for character in FREQUENCIES[language]:
        if not target_have_accents and is_accentuated(character):
            target_have_accents = True
        if target_pure_latin and is_latin(character) is False:
            target_pure_latin = False

    return target_have_accents, target_pure_latin


def alphabet_languages(
    characters: List[str], ignore_non_latin: bool = False
) -> List[str]:
    """
    Return associated languages associated to given characters.
    """
    languages = []  # type: List[Tuple[str, float]]

    source_have_accents = any(is_accentuated(character) for character in characters)

    for language, language_characters in FREQUENCIES.items():

        target_have_accents, target_pure_latin = get_target_features(language)

        if ignore_non_latin and target_pure_latin is False:
            continue

        if target_have_accents is False and source_have_accents:
            continue

        character_count = len(language_characters)  # type: int

        character_match_count = len(
            [c for c in language_characters if c in characters]
        )  # type: int

        ratio = character_match_count / character_count  # type: float

        if ratio >= 0.2:
            languages.append((language, ratio))

    languages = sorted(languages, key=lambda x: x[1], reverse=True)

    return [compatible_language[0] for compatible_language in languages]


def characters_popularity_compare(
    language: str, ordered_characters: List[str]
) -> float:
    """
    Determine if a ordered characters list (by occurrence from most appearance to rarest) match a particular language.
    The result is a ratio between 0. (absolutely no correspondence) and 1. (near perfect fit).
    Beware that is function is not strict on the match in order to ease the detection. (Meaning close match is 1.)
    """
    if language not in FREQUENCIES:
        raise ValueError("{} not available".format(language))

    character_approved_count = 0  # type: int

    for character in ordered_characters:
        if character not in FREQUENCIES[language]:
            continue

        characters_before_source = FREQUENCIES[language][
            0 : FREQUENCIES[language].index(character)
        ]  # type: List[str]
        characters_after_source = FREQUENCIES[language][
            FREQUENCIES[language].index(character) :
        ]  # type: List[str]

        characters_before = ordered_characters[
            0 : ordered_characters.index(character)
        ]  # type: List[str]
        characters_after = ordered_characters[
            ordered_characters.index(character) :
        ]  # type: List[str]

        before_match_count = [
            e in characters_before for e in characters_before_source
        ].count(
            True
        )  # type: int
        after_match_count = [
            e in characters_after for e in characters_after_source
        ].count(
            True
        )  # type: int

        if len(characters_before_source) == 0 and before_match_count <= 4:
            character_approved_count += 1
            continue

        if len(characters_after_source) == 0 and after_match_count <= 4:
            character_approved_count += 1
            continue

        if (
            before_match_count / len(characters_before_source) >= 0.4
            or after_match_count / len(characters_after_source) >= 0.4
        ):
            character_approved_count += 1
            continue

    return character_approved_count / len(ordered_characters)


def alpha_unicode_split(decoded_sequence: str) -> List[str]:
    """
    Given a decoded text sequence, return a list of str. Unicode range / alphabet separation.
    Ex. a text containing English/Latin with a bit a Hebrew will return two items in the resulting list;
    One containing the latin letters and the other hebrew.
    """
    layers = OrderedDict()  # type: Dict[str, str]

    for character in decoded_sequence:
        if character.isalpha() is False:
            continue

        character_range = unicode_range(character)  # type: Optional[str]

        if character_range is None:
            continue

        layer_target_range = None  # type: Optional[str]

        for discovered_range in layers:
            if (
                is_suspiciously_successive_range(discovered_range, character_range)
                is False
            ):
                layer_target_range = discovered_range
                break

        if layer_target_range is None:
            layer_target_range = character_range

        if layer_target_range not in layers:
            layers[layer_target_range] = character.lower()
            continue

        layers[layer_target_range] += character.lower()

    return list(layers.values())


def merge_coherence_ratios(results: List[CoherenceMatches]) -> CoherenceMatches:
    """
    This function merge results previously given by the function coherence_ratio.
    The return type is the same as coherence_ratio.
    """
    per_language_ratios = OrderedDict()  # type: Dict[str, List[float]]
    for result in results:
        for sub_result in result:
            language, ratio = sub_result
            if language not in per_language_ratios:
                per_language_ratios[language] = [ratio]
                continue
            per_language_ratios[language].append(ratio)

    merge = [
        (
            language,
            round(
                sum(per_language_ratios[language]) / len(per_language_ratios[language]),
                4,
            ),
        )
        for language in per_language_ratios
    ]

    return sorted(merge, key=lambda x: x[1], reverse=True)


@lru_cache(maxsize=2048)
def coherence_ratio(
    decoded_sequence: str, threshold: float = 0.1, lg_inclusion: Optional[str] = None
) -> CoherenceMatches:
    """
    Detect ANY language that can be identified in given sequence. The sequence will be analysed by layers.
    A layer = Character extraction by alphabets/ranges.
    """

    results = []  # type: List[Tuple[str, float]]
    ignore_non_latin = False  # type: bool

    sufficient_match_count = 0  # type: int

    lg_inclusion_list = lg_inclusion.split(",") if lg_inclusion is not None else []
    if "Latin Based" in lg_inclusion_list:
        ignore_non_latin = True
        lg_inclusion_list.remove("Latin Based")

    for layer in alpha_unicode_split(decoded_sequence):
        sequence_frequencies = Counter(layer)  # type: Counter
        most_common = sequence_frequencies.most_common()

        character_count = sum(o for c, o in most_common)  # type: int

        if character_count <= TOO_SMALL_SEQUENCE:
            continue

        popular_character_ordered = [c for c, o in most_common]  # type: List[str]

        for language in lg_inclusion_list or alphabet_languages(
            popular_character_ordered, ignore_non_latin
        ):
            ratio = characters_popularity_compare(
                language, popular_character_ordered
            )  # type: float

            if ratio < threshold:
                continue
            elif ratio >= 0.8:
                sufficient_match_count += 1

            results.append((language, round(ratio, 4)))

            if sufficient_match_count >= 3:
                break

    return sorted(results, key=lambda x: x[1], reverse=True)

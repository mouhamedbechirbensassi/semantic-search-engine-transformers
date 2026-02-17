from __future__ import annotations

import ast
import re
from typing import Any

_whitespace_re = re.compile(r"\s+")
_control_re = re.compile(r"[\x00-\x1f\x7f]")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = _control_re.sub(" ", s)
    s = _whitespace_re.sub(" ", s).strip()
    return s


def _try_parse_list_string(value: str) -> Any:
    """
    Try to convert strings like "['A', 'B']" into a real Python list safely.
    If parsing fails, return the original string.
    """
    v = value.strip()
    if (v.startswith("[") and v.endswith("]")) or (v.startswith("(") and v.endswith(")")):
        try:
            return ast.literal_eval(v)
        except Exception:
            return value
    return value


def join_list(value: Any, sep: str = ", ") -> str:
    if value is None:
        return ""

    # If it's a string that looks like a list, try parsing it
    if isinstance(value, str):
        value = _try_parse_list_string(value)

    if isinstance(value, list):
        return sep.join([normalize_text(x) for x in value if x is not None])

    # Handle "[]" or empty
    s = normalize_text(value)
    if s in ("[]", ""):
        return ""
    return s

def scrub_nan_tokens(text: Any) -> str:
    """
    Remove literal 'nan' / 'NaN' tokens that can appear as text.
    Keeps real words like 'Nana' untouched (token-based removal).
    """
    s = normalize_text(text)
    # Remove standalone nan tokens only
    s = re.sub(r"\bNaN\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


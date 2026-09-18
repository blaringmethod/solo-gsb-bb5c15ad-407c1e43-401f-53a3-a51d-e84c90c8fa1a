"""RFC 8785 (JSON Canonicalization Scheme, JCS) serialization.

Turns a parsed JSON value into the deterministic, whitespace-free text
RFC 8785 defines:

* object keys are sorted by their UTF-16 code units (not code points);
* numbers use the ECMAScript ``Number::toString`` format, built on top of
  Python's shortest round-trip ``repr``;
* strings use only the mandatory escapes (``\\"``, ``\\\\``, ``\\b``,
  ``\\t``, ``\\n``, ``\\f``, ``\\r`` and ``\\u00xx`` for the remaining
  control characters); every other character, including non-ASCII, is
  emitted as-is;
* literals and array order follow JSON semantics.
"""

from __future__ import annotations

import math
from typing import Any


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented in canonical form."""


_STRING_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _escape_string(value: str) -> str:
    """Escape a string with only the escapes RFC 8785 mandates."""
    parts = ['"']
    for ch in value:
        escape = _STRING_ESCAPES.get(ch)
        if escape is not None:
            parts.append(escape)
            continue
        code = ord(ch)
        if code < 0x20:
            parts.append(f"\\u{code:04x}")
        elif 0xD800 <= code <= 0xDFFF:
            raise CanonicalizationError("lone surrogate in string")
        else:
            parts.append(ch)
    parts.append('"')
    return "".join(parts)


def _utf16_sort_key(key: str) -> bytes:
    """Sort key giving the UTF-16 code unit order RFC 8785 requires."""
    try:
        return key.encode("utf-16-be")
    except UnicodeEncodeError:
        raise CanonicalizationError("lone surrogate in object key") from None


def _number_to_string(value: float) -> str:
    """Format a float per ECMAScript ``Number::toString``.

    ``repr`` already yields the shortest decimal string that round-trips
    to ``value``. From its digits ``s`` (``k`` of them) and ``n`` (the
    decimal point position: value lies in ``[10**(n-1), 10**n)``) the
    ECMAScript rules pick one of four layouts.
    """
    if not math.isfinite(value):
        raise CanonicalizationError("number outside the IEEE 754 double range")
    if value == 0:  # also folds -0.0 into "0"
        return "0"
    sign = ""
    if value < 0:
        sign = "-"
        value = -value
    mantissa, _, exponent = repr(value).partition("e")
    int_part, _, frac_part = mantissa.partition(".")
    digits = (int_part + frac_part).lstrip("0")
    n = (int(exponent) if exponent else 0) - len(frac_part) + len(digits)
    digits = digits.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    if k > 1:
        digits = digits[0] + "." + digits[1:]
    return f"{sign}{digits}e{'+' if n > 1 else '-'}{abs(n - 1)}"


def canonicalize(value: Any) -> str:
    """Serialize a parsed JSON value to its RFC 8785 canonical text."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            raise CanonicalizationError(
                "number outside the IEEE 754 double range"
            ) from None
        return _number_to_string(number)
    if isinstance(value, list):
        return "[" + ",".join(canonicalize(item) for item in value) + "]"
    if isinstance(value, dict):
        members = [
            _escape_string(key) + ":" + canonicalize(value[key])
            for key in sorted(value, key=_utf16_sort_key)
        ]
        return "{" + ",".join(members) + "}"
    raise CanonicalizationError(f"unsupported value type: {type(value).__name__}")

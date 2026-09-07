import re
import unicodedata

PREPROC_VERSION = "1.2.0"

_TATWEEL = "ـ"
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_PHONE_RE = re.compile(r"(?:\+?966|0)5\d{8}")
_NATIONAL_ID_RE = re.compile(r"\b[12]\d{9}\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_TATWEEL, "")
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def mask_pii(text: str) -> str:
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _NATIONAL_ID_RE.sub("<NATIONAL_ID>", text)
    return text


def preprocess(text: str) -> str:
    text = mask_pii(text)
    text = normalize(text)
    return text

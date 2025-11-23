# src/utils/answer_extraction.py
# -*- coding: utf-8 -*-
import re
from typing import Optional

_SCI_RE = r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
_NUM_RE = rf'{_SCI_RE}'
_BOX_RE = re.compile(r'\\boxed\{([^{}]*)\}')
_HASH_RE = re.compile(r'####\s*(' + _NUM_RE + r')')
_IMAG_UNIT_RE = re.compile(r'^([+-]?)(?:([0-9]+(?:\.[0-9]+)?)?)i$')

def _extract_boxed_segments(text: Optional[str]) -> list:
    segments = []
    if not isinstance(text, str):
        return segments
    needle = r"\boxed{"
    idx = 0
    length = len(text)
    while True:
        start = text.find(needle, idx)
        if start == -1:
            break
        i = start + len(needle)
        depth = 1
        content_chars = []
        while i < length and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
                content_chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
                content_chars.append(ch)
            else:
                content_chars.append(ch)
            i += 1
        if depth == 0:
            segments.append("".join(content_chars).strip())
        idx = i
    return segments

def _normalize_number(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()

    # Handle escaped backslashes from JSON strings (e.g., "\\dfrac" -> "\dfrac")
    s = s.replace('\\\\', '\\')

    s = s.replace(',', '')
    s = s.replace('$', '')
    s = s.replace('%', '')
    s = s.replace(' ', '')

    # Support both \frac and \dfrac
    m = re.fullmatch(r'(-?)\\d?frac\{\s*([0-9]+)\s*\}\{\s*([0-9]+)\s*\}', s)
    if m:
        sign = "-" if m.group(1) == "-" else ""
        a = int(m.group(2)); b = int(m.group(3)) if int(m.group(3)) != 0 else 1
        val = a / b
        out = f"{val:.10f}".rstrip("0").rstrip(".")
        return sign + (out if out else "0")

    m2 = re.fullmatch(_NUM_RE, s)
    if m2:
        try:
            v = float(s)
            if abs(v - int(v)) < 1e-9:
                return str(int(v))
            return str(v)
        except Exception:
            return s

    m_imag = _IMAG_UNIT_RE.fullmatch(s)
    if m_imag:
        sign_token = m_imag.group(1) or ""
        coef_raw = m_imag.group(2)

        # 处理 "i" 和 "-i"
        if not coef_raw or coef_raw == "1":
            return "i" if sign_token != "-" else "-i"
        
        # 处理 "2i", "-3.14i" 等
        coef_norm = _normalize_number(coef_raw)
        if coef_norm is None: # 如果系数无法标准化，返回原始系数
            coef_norm = coef_raw
        
        # 统一处理符号
        if coef_norm.startswith("-"):
            coef_norm = coef_norm[1:]
            sign_token = "-" if sign_token != "-" else ""
        
        if coef_norm in ("1", "+1"):
            return "i" if sign_token != "-" else "-i"

        prefix = "-" if sign_token == "-" else ""
        return f"{prefix}{coef_norm}i"

    # Try to evaluate simple arithmetic expressions
    if re.fullmatch(r'[0-9+\-*/().\s]+', s):
        try:
            calculated = eval(s, {"__builtins__": None}, {})
            if isinstance(calculated, (int, float)):
                return _normalize_number(str(calculated))
        except Exception:
            pass

    nums = re.findall(_NUM_RE, s)
    if nums:
        return _normalize_number(nums[-1])
    return None

def extract_numeric_answer(text: str, dataset_hint: Optional[str] = None) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None

    hint = (dataset_hint or "").lower()

    if "math" in hint:
        # Common patterns in math_dataset: "Answer: ####", "Final answer:", etc.
        answer_line = re.search(r'(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)', text, flags=re.IGNORECASE)
        if answer_line:
            candidate = answer_line.group(1)
            normalized = _normalize_number(candidate)
            if normalized is not None:
                return normalized
            # Try extracting expression inside braces then evaluating
            expr_candidate = re.search(r'\{([^{}]+)\}', candidate)
            if expr_candidate:
                expr_text = expr_candidate.group(1)
                try:
                    value = eval(expr_text, {"__builtins__": None}, {})
                    normalized = _normalize_number(str(value))
                    if normalized is not None:
                        return normalized
                except Exception:
                    pass

    if "svamp" in hint:
        colon_match = re.search(r'[:=]\s*([0-9\s\./-]+)$', text.strip())
        if colon_match:
            normalized = _normalize_number(colon_match.group(1))
            if normalized is not None:
                return normalized
        word_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12"
        }
        for word, num in word_map.items():
            if re.search(fr'\b{word}\b', text, flags=re.IGNORECASE):
                return num

    if hint.startswith("gsm"):
        m = _HASH_RE.search(text)
        if m:
            return _normalize_number(m.group(1))

    boxed_segments = _extract_boxed_segments(text)
    for segment in reversed(boxed_segments):
        normalized = _normalize_number(segment)
        if normalized is not None:
            return normalized

    m = _HASH_RE.search(text)
    if m:
        return _normalize_number(m.group(1))

    # Check if the entire text is an imaginary number before looking for regular numbers
    # This prevents "2i" from being extracted as "2" instead of "2i"
    text_stripped = text.strip()
    if _IMAG_UNIT_RE.fullmatch(text_stripped):
        return _normalize_number(text_stripped)

    nums = re.findall(_NUM_RE, text)
    if nums:
        # Prefer the last purely numeric token
        numeric_tokens = [n for n in nums if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', n.strip())]
        if numeric_tokens:
            return _normalize_number(numeric_tokens[-1])
        # Fallback: try to evaluate simple arithmetic expressions
        for candidate in reversed(nums):
            cleaned = candidate.strip()
            if re.fullmatch(r'[0-9+\-*/().\s]+', cleaned):
                try:
                    value = eval(cleaned, {"__builtins__": None}, {})
                    return _normalize_number(str(value))
                except Exception:
                    continue
        return _normalize_number(nums[-1])

    # Final fallback: try to normalize the entire text (handles cases like "i", "-i", "2i", etc.)
    normalized = _normalize_number(text.strip())
    if normalized is not None:
        return normalized

    return None
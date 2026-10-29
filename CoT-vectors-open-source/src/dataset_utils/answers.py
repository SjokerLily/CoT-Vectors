"""Answer extraction shared by the main reasoning datasets (no model dependencies)."""
import re


def boxed_answer(text):
    """Return the last balanced boxed expression, including nested LaTeX braces."""
    text = str(text)
    result = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    result = text[start:index].strip()
                    break
    return result


def math_answer(text):
    text = str(text).strip()
    boxed = boxed_answer(text)
    if boxed is not None:
        return boxed
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    return text


def numeric_answer(text):
    text = str(text).strip()
    explicit = boxed_answer(text)
    if explicit is not None:
        return explicit
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    return numbers[-1] if numbers else text


def choice_answer(text, choices="ABCDEFGHIJ"):
    text = str(text).strip()
    explicit = boxed_answer(text)
    if explicit is not None:
        text = explicit.strip()
    if re.fullmatch("[" + choices + "]", text):
        return text
    matches = re.findall(r"(?:answer\s*(?:is|:)|choice\s*(?:is|:))\s*\(?([" + choices + r"])\)?", text, re.I)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\b([" + choices + r"])\b", text)
    return matches[-1] if matches else None


def boolean_answer(text):
    if isinstance(text, bool):
        return str(text)
    text = str(text).strip()
    explicit = boxed_answer(text)
    if explicit is not None:
        text = explicit
    matches = re.findall(r"\b(true|false|yes|no)\b", text, re.I)
    if not matches:
        return None
    return "True" if matches[-1].lower() in ("true", "yes") else "False"


def answers_equal(prediction, target):
    return prediction is not None and target is not None and str(prediction).strip() != "" and prediction == target

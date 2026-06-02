from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytesseract
import cv2


DATE_RE = re.compile(r"\b([0-3]?\d[/-][01]?\d[/-](?:19|20)?\d{2})\b")
CPF_RE = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b")
REGISTRO_RE = re.compile(r"\b(\d{9,11})\b")
LETTER_RE = r"A-Za-z\u00C0-\u00FF"


CNH_MARKERS = {
    "titulo_cnh": (r"CARTEIRA\s+NACIONAL\s+DE\s+HABILITACAO", 4),
    "driver_license": (r"DRIVER\s+LICENSE", 1),
    "permiso_conduccion": (r"PERMISO\s+DE\s+CONDUC", 1),
    "territorio_nacional": (r"VALID[AO]?\s+EM\s+TODO\s+O\s+TERRITORIO\s+NACIONAL", 4),
    "republica_brasil": (r"REPUBLICA\s+FEDERATIVA\s+DO\s+BRASIL", 2),
    "senatran": (r"SECRETARIA\s+NACIONAL\s+DE\s+TRANSITO|SENATRAN", 2),
    "campos_cnh": (r"NACIONALIDADE|CPF|CAT\s*HAB|N\s*REGISTRO|DOC\s*IDENTIDADE", 1),
}

CNH_FRONT_ROIS = {
    "cpf": (0.455, 0.545, 0.205, 0.085),
    "numero_registro": (0.655, 0.545, 0.190, 0.085),
    "nacionalidade": (0.455, 0.565, 0.525, 0.060),
}


@dataclass
class CNHData:
    documento_cnh: bool | None = None
    nome: str | None = None
    data_nascimento: str | None = None
    nacionalidade: str | None = None
    local_nascimento: str | None = None
    cpf: str | None = None
    documento_identidade: str | None = None
    orgao_emissor: str | None = None
    uf: str | None = None

    confianca_campos: dict[str, float] = field(default_factory=dict)
    marcadores_cnh: list[str] = field(default_factory=list)
    campos_detectados: dict[str, str] = field(default_factory=dict)


FIELD_LABELS = {
    "nome": ["NOME E SOBRENOME", "NOME"],
    "data_nascimento": ["DATA NASCIMENTO", "DATA DE NASCIMENTO", "NASCIMENTO"],
    "nacionalidade": ["NACIONALIDADE"],
    "local_nascimento": ["LOCAL", "LOCAL NASCIMENTO", "LOCAL DE NASCIMENTO", "NATURALIDADE"],
    "cpf": ["CPF"],
    "documento_identidade": ["DOC IDENTIDADE", "DOCUMENTO IDENTIDADE", "IDENTIDADE", "RG"],
    "orgao_emissor": ["ORG EMISSOR", "ORGAO EMISSOR", "ÓRGÃO EMISSOR", "EMISSOR"],
    "uf": ["UF"],
}



# Normalização de texto


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    match = DATE_RE.search(value)
    if match:
        value = match.group(1)
    elif not re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", value.strip()):
        return clean_field(value)
    parts = re.split(r"[/-]", value)
    if len(parts[-1]) == 2:
        year = int(parts[-1])
        parts[-1] = str(2000 + year if year < 40 else 1900 + year)
    return "/".join(part.zfill(2) if i < 2 else part for i, part in enumerate(parts))


def clean_field(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(rf"[^{LETTER_RE}0-9 ,./'-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" :-")
    return value or None


def clean_letters(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(rf"[^{LETTER_RE} ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def clean_digits(value: str | None) -> str | None:
    digits = only_digits(value)
    return digits or None


def clean_uf(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\b([A-Z]{2})\b", value.upper())
    return match.group(1) if match else None


def normalize_nacionalidade(value: str | None) -> str | None:
    if not value:
        return None

    upper = normalize_for_match(value)

    # Correções comuns do OCR
    upper = upper.replace("8RASILEIRO", "BRASILEIRO")
    upper = upper.replace("BRASILEIR0", "BRASILEIRO")
    upper = upper.replace("BRASILE1RO", "BRASILEIRO")
    upper = upper.replace("BRASILElRO", "BRASILEIRO")

    # Aceita somente se a palavra esperada aparecer no texto
    if re.search(r"\bBRASILEIR[OA]\b", upper):
        return "BRASILEIRO(A)"

    return None


def useful_field(value: str | None) -> str | None:
    value = clean_field(value)
    if not value or len(re.findall(rf"[{LETTER_RE}]", value)) < 3:
        return None
    return value


def is_probable_label_text(value: str | None) -> bool:
    if not value:
        return True
    upper = normalize_for_match(value)
    label_hits = (
        "REPUBLICA", "FEDERATIVA", "BRASIL", "MINISTERIO", "SECRETARIA",
        "CARTEIRA", "DRIVER", "LICENSE", "PERMISO", "CONDUCIR", "REGISTRO",
        "CPF", "NOME", "NASCIMENTO", "VALIDADE", "IDENTIDADE", "EMISSOR",
    )
    if any(hit in upper for hit in label_hits):
        return True
    if re.search(r"HABILITACA[OQ0]|HABILITA", upper):
        return True
    return False


def only_digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")



# CPF — validação matemática completa


def validate_cpf_digits(cpf: str | None) -> bool:
    """Valida os dois dígitos verificadores pelo algoritmo oficial da Receita Federal.
    """
    digits = only_digits(cpf)
    if len(digits) != 11 or len(set(digits)) == 1:
        return False

    def _check(digits: str, length: int) -> bool:
        weights = range(length + 1, 1, -1)
        total = sum(int(d) * w for d, w in zip(digits[:length], weights))
        remainder = (total * 10) % 11
        expected = 0 if remainder == 10 else remainder
        return int(digits[length]) == expected

    return _check(digits, 9) and _check(digits, 10)


def normalize_cpf(value: str | None) -> str | None:

    digits = only_digits(value)
    if len(digits) != 11:
        return None
    formatted = f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"
    if not validate_cpf_digits(formatted):
        return None
    return formatted


def normalize_cpf_from_words(value: str | None) -> str | None:
    """Busca uma sequência com aparência de CPF em texto livre e valida matematicamente."""
    for match in re.finditer(r"\d[\d.\-\s]{9,}\d", value or ""):
        cpf = normalize_cpf(match.group(0))
        if cpf:
            return cpf
    return None



# Outros normalizadores


def normalize_registro_cnh(value: str | None) -> str | None:
    digits = only_digits(value)
    if len(digits) == 11:
        return digits
    if 9 <= len(digits) <= 10 and value and not is_probable_label_text(value):
        return digits
    return None



# Pré-processamento de imagem


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    scale = 2 if max(gray.shape) < 1800 else 1
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9,
    )


def preprocess_roi_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold



# Recorte de página e frente da CNH


def crop_probable_cnh_page(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    left_half = image[:, : int(width * 0.55)]
    gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 210, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > width * height * 0.05 and h > height * 0.25 and w > width * 0.15:
            candidates.append((x, y, w, h))

    if not candidates:
        return image

    x, y, w, h = max(candidates, key=lambda box: box[2] * box[3])
    pad = 18
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, left_half.shape[1])
    y1 = min(y + h + pad, height)
    return left_half[y0:y1, x0:x1]


def crop_front_side(cnh_page: np.ndarray) -> np.ndarray:
    height, width = cnh_page.shape[:2]
    if height / max(width, 1) > 1.15:
        return cnh_page[: int(height * 0.58), :]
    return cnh_page



# OCR


def ocr_image(image: np.ndarray, lang: str) -> str:
    processed = preprocess_for_ocr(image)
    config = "--oem 3 --psm 6"
    return normalize_text(pytesseract.image_to_string(processed, lang=lang, config=config))


def ocr_image_with_confidence(
    image: np.ndarray,
    lang: str,
) -> tuple[str, dict[str, float]]:

    processed = preprocess_for_ocr(image)
    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        processed, lang=lang, config=config,
        output_type=pytesseract.Output.DICT,
    )

    # Índice palavra → confiança (mantém o máximo para palavras repetidas).
    word_confidences: dict[str, float] = {}

    lines_map: dict[tuple[int, int, int], list[str]] = {}

    for i, word in enumerate(data.get("text", [])):
        word = (word or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf < 0:
            continue

        norm = normalize_for_match(word)
        if norm:
            word_confidences[norm] = max(word_confidences.get(norm, 0.0), conf)

        line_key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        lines_map.setdefault(line_key, []).append(word)

    text_lines: list[str] = []
    for key in sorted(lines_map):
        cleaned = clean_field(" ".join(lines_map[key]))
        if cleaned:
            text_lines.append(cleaned)

    full_text = normalize_text("\n".join(text_lines))
    return full_text, word_confidences


def _field_confidence(value: str | None, word_confidences: dict[str, float]) -> float:

    if not value:
        return 0.0

    # Divide em tokens normalizados, eliminando tokens vazios.
    tokens = [t for t in normalize_for_match(value).split() if t]
    if not tokens:
        return 0.0

    found_confs: list[float] = []
    for token in tokens:
        if token in word_confidences:
            found_confs.append(word_confidences[token])
        else:
     
            for indexed_word, conf in word_confidences.items():
                if token in indexed_word or indexed_word in token:
                    found_confs.append(conf)
                    break

    if not found_confs:
        return 0.0

    return round(sum(found_confs) / len(found_confs), 2)


def ocr_for_document_detection(image: np.ndarray, lang: str) -> str:
    page = crop_probable_cnh_page(image)
    front = crop_front_side(page)
    variants = [
        page,
        front,
        cv2.rotate(front, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(front, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    chunks: list[str] = []
    for variant in variants:
        chunks.append(ocr_image(variant, lang=lang))
    return normalize_text("\n".join(chunks))


def ocr_words(image: np.ndarray, lang: str = "por+eng") -> list[dict[str, Any]]:
    processed = preprocess_for_ocr(image)
    data = pytesseract.image_to_data(
        processed, lang=lang, config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )
    words: list[dict[str, Any]] = []
    for index, text in enumerate(data.get("text", [])):
        text = clean_field(text)
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            confidence = -1.0
        if confidence < 0:
            continue
        words.append({
            "text": text,
            "left": int(data["left"][index]),
            "top": int(data["top"][index]),
            "width": int(data["width"][index]),
            "height": int(data["height"][index]),
            "conf": confidence,
        })
    return words


def ocr_line_with_confidence(
    image: np.ndarray,
    lang: str = "por+eng",
    whitelist: str | None = None,
) -> tuple[str, float]:
    processed = preprocess_roi_for_ocr(image)
    config = "--oem 3 --psm 7"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(
        processed, lang=lang, config=config,
        output_type=pytesseract.Output.DICT,
    )
    texts: list[str] = []
    confidences: list[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        cleaned = clean_field(text)
        if not cleaned:
            continue
        try:
            conf_value = float(conf)
        except ValueError:
            continue
        if conf_value >= 0:
            texts.append(cleaned)
            confidences.append(conf_value)
    final_text = normalize_text(" ".join(texts))
    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    return final_text, avg_conf



# Recortes posicionais por rótulo


def crop_relative(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, w, h = box
    x0 = max(int(width * x), 0)
    y0 = max(int(height * y), 0)
    x1 = min(int(width * (x + w)), width)
    y1 = min(int(height * (y + h)), height)
    return image[y0:y1, x0:x1]


def crop_near_word(
    image: np.ndarray,
    words: list[dict[str, Any]],
    label_pattern: str,
    relative_fallback: tuple[float, float, float, float],
    x_offset: float,
    width_factor: float,
) -> np.ndarray:
    label_re = re.compile(label_pattern, re.IGNORECASE)
    candidates = [word for word in words if label_re.search(word["text"])]
    if not candidates:
        return crop_relative(image, relative_fallback)

    label = max(candidates, key=lambda word: (word["top"], word["left"]))
    height, width = image.shape[:2]
    line_height = max(label["height"], int(height * 0.035))
    x0 = int(label["left"] + label["width"] * x_offset)
    y0 = int(label["top"] - line_height * 0.6)
    x1 = int(x0 + label["width"] * width_factor)
    y1 = int(label["top"] + line_height * 2.6)

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(max(x1, x0 + 10), width)
    y1 = min(max(y1, y0 + 10), height)
    return image[y0:y1, x0:x1]


def crop_below_word(
    image: np.ndarray,
    words: list[dict[str, Any]],
    label_pattern: str,
    relative_fallback: tuple[float, float, float, float],
    x_margin: float = 0.02,
    y_gap: float = 0.002,
    height_ratio: float = 0.085,
) -> np.ndarray:
    label_re = re.compile(label_pattern, re.IGNORECASE)
    candidates = [word for word in words if label_re.search(word["text"])]
    if not candidates:
        return crop_relative(image, relative_fallback)

    label = max(candidates, key=lambda word: (word["top"], word["left"]))
    height, width = image.shape[:2]
    x0 = max(int(label["left"] - width * x_margin), 0)
    y0 = max(int(label["top"] + label["height"] + height * y_gap), 0)
    x1 = min(int(width * 0.985), width)
    y1 = min(int(y0 + height * height_ratio), height)
    return image[y0:y1, x0:x1]



# Extração por layout posicional


def save_debug_crop(debug_dir: Path | None, filename: str, image: np.ndarray) -> None:
    if not debug_dir:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / filename), image)


def extract_layout_fields(
    cnh_page: np.ndarray,
    debug_dir: Path | None = None,
) -> tuple[dict[str, str], dict[str, float]]:
    """Extrai CPF e nacionalidade por posição no layout da CNH.
    """
    front = crop_front_side(cnh_page)
    fields: dict[str, str] = {}
    confidences: dict[str, float] = {}
    words = ocr_words(front)

    # --- CPF ---
    cpf_roi = crop_near_word(
        front, words, r"^CPF$",
        CNH_FRONT_ROIS["cpf"],
        x_offset=1.4, width_factor=7.0,
    )
    save_debug_crop(debug_dir, "cpf_roi.png", cpf_roi)

    cpf_text, cpf_conf = ocr_line_with_confidence(
        cpf_roi, lang="eng", whitelist="0123456789.-",
    )

    # Aceita apenas CPF matematicamente válido; ignora qualquer outra sequência.
    cpf = normalize_cpf(cpf_text) or normalize_cpf_from_words(cpf_text)
    if cpf:
        fields["cpf"] = cpf
        confidences["cpf"] = cpf_conf
    # Se o CPF não passou na validação, o campo simplesmente não é adicionado.
    # confiança permanece ausente (0.0 implícito no merge).

    # --- Nacionalidade ---
    nacionalidade = None
    nacionalidade_conf = 0.0

    # Usa OCR da frente inteira, sem novo recorte, para evitar pegar filiação.
    words_front = ocr_words(front, lang="por+eng")

    candidatos = []
    for word in words_front:
        norm = normalize_for_match(word["text"])

        if re.search(r"BRASILEIR[OA]|BRASILE1R[OA]|BRASILEIR0|BRAS1LEIR[OA]", norm):
            candidatos.append(word)

    if candidatos:
        melhor = max(candidatos, key=lambda w: w["conf"])
        nacionalidade = "BRASILEIRO(A)"
        nacionalidade_conf = float(melhor["conf"])

    if nacionalidade:
        fields["nacionalidade"] = nacionalidade
        confidences["nacionalidade"] = nacionalidade_conf

    return fields, confidences



# Identificação do documento


def identify_cnh_document(image_path: Path, lang: str = "por+eng") -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")
    return identify_cnh_image(image, lang=lang)


def identify_cnh_image(image: np.ndarray, lang: str = "por+eng") -> dict[str, Any]:
    text = ocr_for_document_detection(image, lang=lang)
    normalized = normalize_for_match(text)
    markers: list[str] = []
    score = 0
    for marker, (pattern, weight) in CNH_MARKERS.items():
        if re.search(pattern, normalized):
            markers.append(marker)
            score += weight

    has_core_marker = any(marker in markers for marker in ("titulo_cnh", "territorio_nacional"))
    has_translation_pair = "driver_license" in markers and "permiso_conduccion" in markers
    is_cnh = score >= 4 and (has_core_marker or has_translation_pair)
    return {
        "is_cnh": is_cnh,
        "score": score,
        "markers": markers,
        "text": text,
    }



# Extração por OCR textual


def extract_after_label(lines: list[str], labels: Iterable[str]) -> str | None:
    normalized_labels = [label.upper() for label in labels]
    for index, line in enumerate(lines):
        upper = line.upper()
        if any(label in upper for label in normalized_labels):
            remainder = re.split("|".join(map(re.escape, normalized_labels)), upper, maxsplit=1)[-1]
            if useful_field(remainder) and not looks_like_label(remainder) and not is_probable_label_text(remainder):
                return useful_field(remainder)
            for next_line in lines[index + 1 : index + 4]:
                candidate = useful_field(next_line)
                if candidate and not looks_like_label(candidate) and not is_probable_label_text(candidate):
                    return candidate
    return None


def extract_known_fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for field_name, labels in FIELD_LABELS.items():
        if field_name in {"cpf", "numero_registro"}:
            continue
        value = extract_after_label(lines, labels)
        if value:
            fields[field_name] = value
    return fields


def extract_regex_fields(text: str) -> dict[str, str]:
    """Extrai campos com padrão forte (CPF, datas, categoria).

    O CPF só é aceito se passar na validação matemática dos dígitos verificadores.
    """
    fields: dict[str, str] = {}

    for match in CPF_RE.finditer(text):
        candidate = normalize_cpf(match.group(1))
        if candidate:
            fields["cpf"] = candidate
            break  # usa o primeiro CPF matematicamente válido encontrado

    dates = [normalize_date(match.group(1)) for match in DATE_RE.finditer(text)]
    date_names = ["data_nascimento", "validade", "primeira_habilitacao", "data_emissao"]
    for field_name, value in zip(date_names, dates):
        if value:
            fields.setdefault(field_name, value)

    category_match = re.search(
        r"\bACC?\s*([A-E])\b|\bCAT(?:EGORIA)?\.?\s*([A-E])\b", text, re.IGNORECASE,
    )
    if category_match:
        fields["categoria"] = next(group for group in category_match.groups() if group)

    return fields


def extract_birth_fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in lines:
        if is_probable_label_text(line) and not DATE_RE.search(line):
            continue
        match = re.search(
            rf"{DATE_RE.pattern}\s*,?\s*([{LETTER_RE} ]+?)\s*,\s*([A-Z]{{2}})\b",
            line, flags=re.IGNORECASE,
        )
        if not match:
            continue
        fields["data_nascimento"] = normalize_date(match.group(1))
        fields["local_nascimento"] = clean_letters(match.group(2))
        fields["uf"] = clean_uf(match.group(3))
        break
    return {key: value for key, value in fields.items() if value}


def extract_identity_fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in lines:
        if DATE_RE.search(line):
            continue
        compact = re.sub(r"\s+", " ", line.upper()).strip()
        match = re.search(r"\b(\d[\d.\-/ ]{3,}\d)\s+([A-Z]{2,12})\s+([A-Z]{2})\b", compact)
        if not match:
            continue
        document = clean_digits(match.group(1))
        issuer = clean_letters(match.group(2))
        uf = clean_uf(match.group(3))
        if document and issuer and uf:
            fields["documento_identidade"] = document
            fields["orgao_emissor"] = issuer
            fields["uf"] = uf
            break
    return fields


def looks_like_label(value: str) -> bool:
    labels = ("NOME", "DATA", "LOCAL", "CPF", "DOC")
    upper = value.upper()
    return any(label in upper for label in labels)


def parse_mrz(text: str) -> dict[str, str]:
    lines = [
        re.sub(r"[^A-Z0-9<]", "", line.upper())
        for line in text.splitlines()
        if "<" in line and len(line.strip()) >= 20
    ]
    result: dict[str, str] = {}
    if lines:
        name_line = max(lines, key=lambda line: line.count("<"))
        name_parts = [part for part in name_line.split("<") if part]
        if len(name_parts) >= 2:
            result["nome"] = clean_field(" ".join(name_parts))
    for line in lines:
        match = re.search(r"([0-9]{6})[0-9<][MF<X<]", line)
        if match:
            result["data_nascimento"] = yymmdd_to_date(match.group(1))
            break
    return result


def yymmdd_to_date(value: str) -> str:
    year = int(value[:2])
    month = value[2:4]
    day = value[4:6]
    full_year = 2000 + year if year < 30 else 1900 + year
    return f"{day}/{month}/{full_year}"


def parse_cnh_text(
    text: str,
    line_confidences: dict[str, float] | None = None,
) -> CNHData:
    """Organiza o texto bruto do OCR em campos estruturados e associa confiança.
    """
    lc = line_confidences or {}

    lines = [clean_field(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    joined = "\n".join(lines)
    mrz = parse_mrz(joined)
    fields = extract_known_fields(lines)
    fields.update({key: value for key, value in extract_regex_fields(joined).items() if key not in fields})
    fields.update(extract_birth_fields(lines))
    fields.update(extract_identity_fields(lines))

    nome = fields.get("nome")
    data_nascimento = fields.get("data_nascimento")
    nacionalidade = fields.get("nacionalidade")
    local_nascimento = fields.get("local_nascimento")

    if not data_nascimento:
        dates = [normalize_date(match.group(1)) for match in DATE_RE.finditer(joined)]
        data_nascimento = dates[0] if dates else None

    local_nascimento_inferred = False
    if not local_nascimento:
        for line in lines:
            if re.search(r"\b[A-Z\u00C0-\u00DD ]+,\s*[A-Z]{2}\b", line.upper()) and not is_probable_label_text(line):
                local_nascimento = line
                local_nascimento_inferred = True
                break

    nacionalidade_inferred = False

    # Valores finais após limpeza — usados para buscar confiança.
    nome_final = clean_letters(nome or mrz.get("nome"))
    data_nasc_final = normalize_date(data_nascimento) or mrz.get("data_nascimento")
    nac_final = normalize_nacionalidade(nacionalidade)
    local_final = clean_letters(local_nascimento)
    cpf_final = normalize_cpf(fields.get("cpf"))
    doc_final = clean_digits(fields.get("documento_identidade"))
    emissor_final = clean_letters(fields.get("orgao_emissor"))
    uf_final = clean_uf(fields.get("uf"))

    # Para campos derivados de uma linha que continha rótulo + valor (extract_after_label),
    # a busca no índice usa o valor limpo; para campos de regex que vieram de linhas com
    # múltiplos tokens (CPF, data, RG), busca o match original antes da limpeza.
    cpf_source = fields.get("cpf") or cpf_final
    doc_source = fields.get("documento_identidade") or doc_final
    emissor_source = fields.get("orgao_emissor") or emissor_final
    uf_source = fields.get("uf") or uf_final

    confianca: dict[str, float] = {
        "nome":               _field_confidence(nome_final, lc),
        "data_nascimento":    _field_confidence(data_nascimento, lc),
        "nacionalidade":      _field_confidence(nac_final, lc),
        "local_nascimento":   _field_confidence(local_final, lc),
        "cpf":                _field_confidence(cpf_source, lc),
        "documento_identidade": _field_confidence(doc_source, lc),
        "orgao_emissor":      _field_confidence(emissor_source, lc),
        "uf":                 _field_confidence(uf_source, lc),
    }

    return CNHData(
        nome=nome_final,
        data_nascimento=data_nasc_final,
        nacionalidade=nac_final,
        local_nascimento=local_final,
        cpf=cpf_final,
        documento_identidade=doc_final,
        orgao_emissor=emissor_final,
        uf=uf_final,
        confianca_campos=confianca,
    )



# Sanitização e mesclagem


def sanitize_cnh_data(data: CNHData) -> CNHData:
    data.nome = clean_letters(data.nome)
    data.nacionalidade = normalize_nacionalidade(data.nacionalidade)
    data.local_nascimento = clean_letters(data.local_nascimento)
    data.orgao_emissor = clean_letters(data.orgao_emissor)
    data.documento_identidade = clean_digits(data.documento_identidade)
    data.uf = clean_uf(data.uf)
    data.cpf = normalize_cpf(data.cpf)  # última barreira: rejeita se inválido
    return data


def merge_data(primary: CNHData, fallback: CNHData) -> CNHData:
    primary_dict = asdict(primary)
    fallback_dict = asdict(fallback)
    merged: dict[str, Any] = {}
    for key, fallback_value in fallback_dict.items():
        primary_value = primary_dict.get(key)
        if isinstance(primary_value, dict) or isinstance(fallback_value, dict):
            merged[key] = {**(fallback_value or {}), **(primary_value or {})}
        elif isinstance(primary_value, list) or isinstance(fallback_value, list):
            merged[key] = primary_value or fallback_value or []
        else:
            merged[key] = primary_value or fallback_value

    if isinstance(primary_value, dict) or isinstance(fallback_value, dict):
        merged[key] = {**(fallback_value or {}), **(primary_value or {})}
    return CNHData(**merged)



# Fluxo principal de extração


def extract_ecnh(
    image_path: Path,
    lang: str = "por+eng",
    debug_dir: Path | None = None,
    validate_document: bool = True,
) -> tuple[CNHData, str]:

 
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

    document_check = identify_cnh_image(image, lang=lang)
    if validate_document and not document_check["is_cnh"]:
        markers = ", ".join(document_check["markers"]) or "nenhum marcador forte"
        raise ValueError(
            "O documento informado não parece ser uma Carteira Nacional de Habilitação. "
            f"Marcadores encontrados: {markers}. "
            "Use validate_document=False ou --skip-validation para recortes parciais."
        )

    cnh_crop = crop_probable_cnh_page(image)
    layout_fields, layout_confidences = extract_layout_fields(cnh_crop, debug_dir=debug_dir)

    # Usa OCR com confiança para obter tanto o texto quanto o índice linha→conf.
    text, line_confidences = ocr_image_with_confidence(cnh_crop, lang=lang)
    ocr_data = parse_cnh_text(text, line_confidences)

    # layout_data tem prioridade para CPF e nacionalidade (recorte posicional mais preciso).
    # A confiança do layout substitui a do OCR textual quando o campo veio do layout.
    layout_data = CNHData(
        documento_cnh=document_check["is_cnh"],
        nacionalidade=layout_fields.get("nacionalidade"),
        cpf=layout_fields.get("cpf"),
        confianca_campos=layout_confidences,
        campos_detectados={f"layout_{key}": value for key, value in layout_fields.items()},
    )

    merged = merge_data(layout_data, ocr_data)
    merged = sanitize_cnh_data(merged)

    # Confiança final: layout prevalece sobre OCR textual para os campos que extraiu.
    # Para os demais campos, usa o que o OCR textual calculou.
    merged_confidences: dict[str, float] = {**ocr_data.confianca_campos}
    for field_name, conf in layout_confidences.items():
        if conf > 0.0:
            merged_confidences[field_name] = conf
    merged.confianca_campos = merged_confidences

    merged.texto_ocr = text
    merged.documento_cnh = document_check["is_cnh"]
    merged.marcadores_cnh = document_check["markers"]
    merged.campos_detectados = {
        **merged.campos_detectados,
        "documento_cnh_score": str(document_check["score"]),
        "documento_cnh_marcadores": ", ".join(document_check["markers"]),
    }
    return merged, text

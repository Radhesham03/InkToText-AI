"""Handwriting recognition: OpenCV line segmentation + pretrained TrOCR."""
import threading
import cv2
import numpy as np
from PIL import Image

MODEL_NAME = "microsoft/trocr-base-handwritten"  # ~1.3 GB, downloaded once, cached by Hugging Face
_lock = threading.Lock()
_model = _processor = _device = None


def load_model():
    """Load TrOCR once and reuse it for every request."""
    global _model, _processor, _device
    with _lock:
        if _model is None:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
            _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(_device).eval()
    return _model, _processor, _device


def model_ready():
    return _model is not None


def _remove_rules(binary):
    """Find long (possibly slightly slanted) ruled lines and margin lines with a Hough transform."""
    h, w = binary.shape
    mask = np.zeros_like(binary)
    found = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=int(w * 0.15),
                            minLineLength=int(min(w, h) * 0.3), maxLineGap=int(w * 0.04))
    if found is not None:
        for x1, y1, x2, y2 in np.asarray(found).reshape(-1, 4):  # works for OpenCV 4 and 5
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang < 8 or ang > 172 or 82 < ang < 98:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 7)
    return mask


def _drop_specks_and_stubs(binary):
    """Remove tiny specks, flat rule stubs and tall margin lines left after rule removal."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    H, W = binary.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < 12:
            continue
        near_edge = x < 0.07 * W or x + cw > 0.93 * W
        if ch <= 6 and cw >= 25 and (near_edge or cw > 0.25 * W):  # flat stub of a ruled line
            continue
        if cw <= 10 and ch > 0.3 * H:  # vertical margin line
            continue
        keep[i] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _scale(bgr):
    """Bring the image to a working size (long side ~1400-1800 px)."""
    h, w = bgr.shape[:2]
    scale = 1800 / max(h, w)
    if scale < 1:  # shrink very large photos for speed
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif max(h, w) < 1400:  # upscale small images so strokes are thick enough
        s = 1400 / max(h, w)
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return bgr


def preprocess(bgr):
    """Grayscale -> denoise -> shadow flattening -> CLAHE -> adaptive threshold -> ruled-line removal.
    Returns (clean_gray_for_model, binary_ink)."""
    bgr = _scale(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((15, 15), np.uint8)), 31)  # flatten uneven lighting
    gray = cv2.divide(gray, bg, scale=255)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    # pass 1: straight rules (morphological opening), pass 2: slanted rules (Hough)
    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, gray.shape[1] // 8), 1))
    flat = cv2.dilate(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kx), np.ones((3, 3), np.uint8))
    binary = cv2.subtract(binary, flat)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    rules = _remove_rules(binary)
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(rules))
    rules = cv2.bitwise_or(rules, flat)
    binary = _drop_specks_and_stubs(binary)
    # note: the model gets the flattened grayscale as-is (TrOCR copes with ruled paper);
    # the rule-free binary is used only to locate the lines.
    return gray, binary


def find_lines(binary):
    """Detect text lines from the smoothed horizontal ink profile. Returns [(y0, y1, x0, x1)]."""
    H, W = binary.shape
    prof = (binary > 0).sum(axis=1).astype(np.float32)
    sigma = max(3.0, H / 120)
    k = np.exp(-0.5 * (np.arange(-int(3 * sigma), int(3 * sigma) + 1) / sigma) ** 2)
    sm = np.convolve(prof, k / k.sum(), mode="same")
    if sm.max() < 3:
        return []
    # local minima (valleys) and maxima (peaks)
    idx = np.arange(1, H - 1)
    minima = [i for i in idx if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1]]
    cuts = [0]
    last_peak = sm[:max(minima[0], 1)].max() if minima else sm.max()
    for j, m in enumerate(minima):
        nxt = minima[j + 1] if j + 1 < len(minima) else H
        left = sm[cuts[-1]:m].max() if m > cuts[-1] else 0
        right = sm[m:nxt].max() if nxt > m else 0
        if sm[m] < 0.55 * min(left, right) and left > 0.12 * sm.max() and right > 0.12 * sm.max():
            cuts.append(m)
    cuts.append(H)
    bands = list(zip(cuts[:-1], cuts[1:]))
    def ink(band):
        return float((binary[band[0]:band[1]] > 0).sum())
    # merge thin bands (dot of an i, split words) and low-ink bands (descenders only) into the band above
    for _ in range(3):
        if len(bands) < 3:
            break
        med_t = float(np.median([b - a for a, b in bands]))
        med_i = float(np.median([ink(bd) for bd in bands]))
        merged = []
        for bd in bands:
            thin = (bd[1] - bd[0]) < 0.6 * med_t
            weak = ink(bd) < 0.35 * med_i
            if (thin or weak) and merged:
                merged[-1] = (merged[-1][0], bd[1])
            else:
                merged.append(bd)
        bands = merged
    lines = []
    for y0, y1 in bands:
        seg = binary[y0:y1]
        if seg.sum() / 255 < max(40, 0.004 * seg.shape[0] * W):
            continue
        colsum = (seg > 0).sum(axis=0).astype(np.float64)
        cum = np.cumsum(colsum) / max(colsum.sum(), 1)
        x0 = int(np.searchsorted(cum, 0.005)); x1 = int(np.searchsorted(cum, 0.995)) + 1
        if x1 - x0 < 15:
            continue
        rows = np.where(seg.sum(axis=1) > 0)[0]
        lines.append((y0 + int(rows[0]), y0 + int(rows[-1]) + 1, x0, min(x1, W)))
    # split very tall bands (merged lines) into equal parts
    if lines:
        med = float(np.median([b - a for a, b, _, _ in lines]))
        out = []
        for y0, y1, x0, x1 in lines:
            n = int(round((y1 - y0) / med)) if (y1 - y0) > 1.8 * med else 1
            step = (y1 - y0) / max(n, 1)
            for i in range(n):
                out.append((int(y0 + i * step), int(y0 + (i + 1) * step), x0, x1))
        lines = out
    return lines


def split_wide(binary_line, x0, x1, height):
    """Cut very wide lines in the middle of a blank gap between words, never inside a word."""
    maxw = max(int(height * 13), 1000)
    cuts, x = [x0], x0
    ink = (binary_line > 0).sum(axis=0)
    while x1 - x > maxw * 1.2:
        lo, hi = x + int(maxw * 0.55) - x0, x + maxw - x0
        win = ink[lo:hi] <= 1
        best, run_start, best_len = None, None, 0
        for i, blank in enumerate(list(win) + [False]):
            if blank and run_start is None:
                run_start = i
            elif not blank and run_start is not None:
                if i - run_start > best_len:
                    best_len, best = i - run_start, run_start + (i - run_start) // 2
                run_start = None
        cut = lo + (best if best is not None else int(np.argmin(ink[lo:hi])))
        x = x0 + cut
        cuts.append(x)
    cuts.append(x1)
    return list(zip(cuts[:-1], cuts[1:]))


def split_bullet(binary_line, x0, line_h):
    """If the line starts with a small bullet/dash mark, return (marker_text, new_x0) so the model never sees it."""
    n, _, st, _ = cv2.connectedComponentsWithStats((binary_line > 0).astype(np.uint8), connectivity=8)
    comps = sorted([tuple(st[i][:4]) for i in range(1, n) if st[i][4] >= 8], key=lambda c: c[0])
    if len(comps) < 2:
        return "", x0
    x, y, w, h = comps[0]
    centre = (y + h / 2) / max(binary_line.shape[0], 1)
    gap = comps[1][0] - (x + w)
    if h < 0.3 * line_h and w < 0.9 * line_h and 0.3 < centre < 0.8 and gap > 0.35 * line_h:
        return "-", x0 + x + w + 2
    return "", x0


def clean_text(t):
    """Tidy TrOCR spacing artefacts: ' ,' -> ',', stray quotes, doubled spaces."""
    import re
    t = re.sub(r"\s+([.,!?;:%])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(^|\s)[\"'`]+(\s|$)", " ", t).strip()
    return t


def _crop_to_pil(gray, y0, y1, x0, x1):
    pad = max(6, int((y1 - y0) * 0.15))
    H, W = gray.shape
    crop = gray[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
    return Image.fromarray(crop).convert("RGB")


# ---------------------------------------------------------------- word-level detector (EasyOCR CRAFT)
_det_lock = threading.Lock()
_detector = None
_det_failed = False


def get_detector():
    """Load EasyOCR's CRAFT text detector once (detection only, no EasyOCR recognizer). None if unavailable."""
    global _detector, _det_failed
    with _det_lock:
        if _detector is None and not _det_failed:
            try:
                import easyocr
                import torch
                _detector = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), recognizer=False, verbose=False)
            except Exception as e:  # noqa
                _det_failed = True
                print("[InkToText] EasyOCR detector unavailable, using line-projection fallback:", e)
    return _detector


def detector_ready():
    return _detector is not None


def warm_up():
    load_model()
    get_detector()


def detect_boxes(det, scaled_bgr):
    """Return word/phrase boxes [(x0, y0, x1, y1)] found by CRAFT."""
    rgb = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2RGB)
    horiz, free = det.detect(rgb, min_size=10, text_threshold=0.6, low_text=0.3,
                             link_threshold=0.3, canvas_size=2560, mag_ratio=1.0)
    H, W = rgb.shape[:2]
    boxes = []
    for x0, x1, y0, y1 in (horiz[0] if horiz else []):
        boxes.append((int(x0), int(y0), int(x1), int(y1)))
    for poly in (free[0] if free else []):
        p = np.asarray(poly)
        boxes.append((int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())))
    out = []
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        w, h = x1 - x0, y1 - y0
        if w < 10 or h < 10 or w * h > 0.5 * W * H:
            continue
        out.append((x0, y0, x1, y1))
    return out


def group_boxes_into_lines(boxes):
    """Cluster boxes into text rows. Walks left to right and compares each box with the nearest box of
    every row, so slightly slanted or curved lines (phone photos) still stay together."""
    glob_h = float(np.median([b[3] - b[1] for b in boxes]))
    rows = []
    for b in sorted(boxes, key=lambda b: b[0]):
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        best, best_d = None, None
        for row in rows:
            prev = [q for q in row if q[0] <= b[0]]
            r = max(prev, key=lambda q: q[2]) if prev else row[0]
            tol = 0.55 * max(h, r[3] - r[1], 0.8 * glob_h)
            d = abs(cy - (r[1] + r[3]) / 2)
            if d < tol and (best_d is None or d < best_d):
                best, best_d = row, d
        if best is None:
            rows.append([b])
        else:
            best.append(b)
    lines = [{"boxes": r, "cy": float(np.mean([(q[1] + q[3]) / 2 for q in r])),
              "h": float(np.median([q[3] - q[1] for q in r]))} for r in rows]
    return sorted(lines, key=lambda L: L["cy"])


def split_row_into_phrases(row_boxes, h):
    """Group neighbouring boxes into phrases (split at big gaps / columns, and when too wide for TrOCR)."""
    bs = sorted(row_boxes, key=lambda b: b[0])
    phrases = [[bs[0]]]
    for b in bs[1:]:
        if b[0] - phrases[-1][-1][2] > 1.3 * h:
            phrases.append([b])
        else:
            phrases[-1].append(b)
    out, maxw = [], max(h * 13, 1000)

    def split(p):
        if len(p) == 1 or p[-1][2] - p[0][0] <= maxw:
            out.append(p)
            return
        gaps = [p[i + 1][0] - p[i][2] for i in range(len(p) - 1)]
        k = int(np.argmax(gaps)) + 1
        split(p[:k]); split(p[k:])

    for p in phrases:
        split(p)
    return out


def _word_lines(det, scaled_bgr, gray):
    """Line items built from CRAFT boxes. Each item: {y0, y1, pieces:[{crop, sep, prefix}]}"""
    boxes = detect_boxes(det, scaled_bgr)
    if len(boxes) < 2:
        return None
    items = []
    for L in group_boxes_into_lines(boxes):
        h = max(L["h"], 12)
        phrases = split_row_into_phrases(L["boxes"], h)
        pieces, prev_x1 = [], None
        for p in phrases:
            x0, x1 = p[0][0], max(q[2] for q in p)
            y0, y1 = min(q[1] for q in p), max(q[3] for q in p)
            gap = 0 if prev_x1 is None else x0 - prev_x1
            sep = "    " if gap > 3 * h else " "   # keep a column-like gap
            pieces.append({"crop": _crop_to_pil(gray, y0, y1, x0, x1), "sep": sep if pieces else "", "prefix": ""})
            prev_x1 = x1
        items.append({"y0": int(min(q[1] for q in L["boxes"])), "y1": int(max(q[3] for q in L["boxes"])), "pieces": pieces})
    return items


# ---------------------------------------------------------------- fallback: projection-profile lines
def _classic_lines(gray, binary):
    items = []
    for y0, y1, x0, x1 in find_lines(binary):
        mark, x0 = split_bullet(binary[y0:y1, x0:x1], x0, y1 - y0)
        pieces = []
        for k, (a, b) in enumerate(split_wide(binary[y0:y1, x0:x1], x0, x1, y1 - y0)):
            pieces.append({"crop": _crop_to_pil(gray, y0, y1, a, b), "sep": " " if k else "",
                           "prefix": mark if k == 0 else ""})
        items.append({"y0": y0, "y1": y1, "pieces": pieces})
    return items


def _ocr_crops(crops):
    import torch
    model, processor, device = load_model()
    texts, confs = [], []
    for i in range(0, len(crops), 8):
        pix = processor(images=crops[i:i + 8], return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(pix, num_beams=4, max_new_tokens=64,
                                 output_scores=True, return_dict_in_generate=True)
        texts += processor.batch_decode(out.sequences, skip_special_tokens=True)
        confs += [float(np.exp(s)) for s in out.sequences_scores.cpu().tolist()]
    return texts, confs


def recognize_image(path, method_pref="auto"):
    """Full pipeline. Returns dict with text, per-line results and metadata."""
    model, processor, device = load_model()
    bgr = cv2.imread(path)
    if bgr is None:
        bgr = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    scaled = _scale(bgr)
    gray, binary = preprocess(scaled)

    items, method = None, "word-detector"
    det = get_detector() if method_pref != "lines" else None
    if det is not None:
        try:
            items = _word_lines(det, scaled, gray)
        except Exception as e:  # noqa
            print("[InkToText] word detector failed, falling back:", e)
    if not items:
        method, items = "line-projection", _classic_lines(gray, binary)
    if not items:
        return {"text": "", "lines": [], "avg_confidence": None, "method": method, "device": device}

    crops = [p["crop"] for it in items for p in it["pieces"]]
    texts, confs = _ocr_crops(crops)

    results, k = [], 0
    for it in items:
        parts, cl = [], []
        for p in it["pieces"]:
            t, c = clean_text(texts[k]), confs[k]; k += 1
            if not t or (c < 0.2 and len(t) <= 3):  # icons, doodles, smudges
                continue
            parts.append((p["sep"], p["prefix"], t)); cl.append(c)
        if not parts:
            continue
        txt = ""
        for i, (sep, prefix, t) in enumerate(parts):
            txt += (sep if i else "") + ((prefix + " ") if prefix else "") + t
        results.append({"text": txt, "confidence": round(float(np.mean(cl)), 3), "y0": it["y0"], "y1": it["y1"]})

    if not results:
        return {"text": "", "lines": [], "avg_confidence": None, "method": method, "device": device}
    med_h = float(np.median([r["y1"] - r["y0"] for r in results]))
    out_lines = []
    for i, r in enumerate(results):
        if i and (r["y0"] - results[i - 1]["y1"]) > 0.9 * med_h:
            out_lines.append("")
        out_lines.append(r["text"])
    return {
        "text": "\n".join(out_lines).strip(),
        "lines": [{"text": r["text"], "confidence": r["confidence"]} for r in results],
        "avg_confidence": round(float(np.mean([r["confidence"] for r in results])), 3),
        "method": method,
        "device": device,
    }

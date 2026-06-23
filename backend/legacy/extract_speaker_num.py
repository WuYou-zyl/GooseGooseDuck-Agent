import cv2
import logging
import numpy as np
import os
import re
import threading
import time
from typing import Optional

logger = logging.getLogger("ggd-a.ocr")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# legacy 在 backend/legacy/ 下，仓库根为再上两级
_REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
template_path = os.path.join(_REPO_ROOT, "template_imgs")

RECOGNITION_MODE = "template"
DEBUG_SAVE = False
DEBUG_DIR = os.path.join(_REPO_ROOT, "_debug_crops")
_templates = None

_ocr_engine = None
_ocr_loading = False
_ocr_ready = threading.Event()
_CHAT_BOX_REL = (0.78, 0.78, 0.22, 0.20)


def enable_debug_save(enabled: bool = True):
    """Turn on/off saving of cropped ROI images for debugging crop regions."""
    global DEBUG_SAVE
    DEBUG_SAVE = enabled
    if enabled:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        logger.info("Debug image saving enabled -> %s", DEBUG_DIR)


def set_recognition_mode(mode: str):
    global RECOGNITION_MODE
    if mode in ["template", "ocr"]:
        RECOGNITION_MODE = mode
    else:
        raise ValueError("mode 必须是 'template' 或 'ocr'")


def _get_ocr(block: bool = False):
    """
    Return the RapidOCR engine if ready, or None if still loading.
    With block=True, waits until loading finishes (used by preload_ocr).
    """
    global _ocr_engine, _ocr_loading
    if _ocr_engine is not None:
        return _ocr_engine
    if not _ocr_loading:
        _ocr_loading = True
        threading.Thread(target=_load_ocr_background, daemon=True).start()
    if block:
        _ocr_ready.wait()
    return _ocr_engine


def _load_ocr_background():
    """Load RapidOCR in a background thread so the monitor loop is never blocked."""
    global _ocr_engine
    import sys
    try:
        print("[OCR] Importing rapidocr_onnxruntime (CPU) ...", flush=True)
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        _ocr_engine = RapidOCR()
        print("[OCR] RapidOCR loaded OK (CPU)", flush=True)
        logger.info("RapidOCR loaded successfully")
    except Exception as e:
        print(f"[OCR] FAILED to load RapidOCR: {e}", file=sys.stderr, flush=True)
        logger.error("Failed to load RapidOCR: %s", e, exc_info=True)
    finally:
        _ocr_ready.set()


def preload_ocr():
    """Kick off OCR loading in background. Non-blocking."""
    _get_ocr(block=False)


def _ocr_collect_lines(img_bgr: np.ndarray) -> str:
    ocr = _get_ocr(block=False)
    if ocr is None or img_bgr is None or img_bgr.size == 0:
        return ""

    h, w = img_bgr.shape[:2]
    scale = max(1, int(720 / max(h, 1)))
    scale = max(scale, 2)
    enlarged = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    if DEBUG_SAVE:
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"chat_crop_{ts}.png"), enlarged)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"chat_bin_{ts}.png"), binary)

    try:
        result, _ = ocr(binary)
    except Exception as e:
        logger.error("Chat-box OCR failed: %s", e)
        return ""
    if not result:
        return ""

    items: list[dict] = []
    for entry in result:
        if not entry or len(entry) < 3:
            continue
        box, text, conf = entry[0], str(entry[1]).strip(), float(entry[2] or 0.0)
        if not text:
            continue
        top_y = min(p[1] for p in box)
        left_x = min(p[0] for p in box)
        items.append({"text": text, "y": top_y, "x": left_x, "conf": conf})
    if not items:
        return ""

    items.sort(key=lambda it: (it["y"], it["x"]))
    line_thresh = max(18, enlarged.shape[0] * 0.08)
    lines: list[list[str]] = []
    current: list[str] = []
    current_y = None
    for item in items:
        if current_y is None or abs(item["y"] - current_y) <= line_thresh:
            current.append(item["text"])
            current_y = item["y"] if current_y is None else (current_y + item["y"]) / 2
        else:
            lines.append(current)
            current = [item["text"]]
            current_y = item["y"]
    if current:
        lines.append(current)

    text = "\n".join(" ".join(parts).strip() for parts in lines if parts).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def extract_chat_box_text(img_bgr: np.ndarray) -> str:
    if img_bgr is None or img_bgr.size == 0:
        return ""
    h, w = img_bgr.shape[:2]
    rx, ry, rw, rh = _CHAT_BOX_REL
    x = max(0, min(int(w * rx), w - 1))
    y = max(0, min(int(h * ry), h - 1))
    cw = max(1, min(int(w * rw), w - x))
    ch = max(1, min(int(h * rh), h - y))
    crop = img_bgr[y : y + ch, x : x + cw]
    return _ocr_collect_lines(crop)


def _ocr_extract_speaker(img_bgr: np.ndarray) -> Optional[str]:
    """
    Run OCR on a downscaled frame to find "X发言中" text anywhere on screen.
    Downscaling to ~640px wide dramatically speeds up OCR (3-5s → <1s).
    """
    ocr = _get_ocr(block=False)
    if ocr is None:
        return None

    h, w = img_bgr.shape[:2]
    # Downscale large frames for speed: keep width ≤ 640px
    max_w = 640
    if w > max_w:
        scale = max_w / w
        img_small = cv2.resize(img_bgr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr

    if DEBUG_SAVE:
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"frame_{ts}.png"), img_small)

    t0 = time.time()
    try:
        result, _ = ocr(img_small)
    except Exception as e:
        logger.error("OCR predict failed: %s", e)
        return None
    elapsed_ms = int((time.time() - t0) * 1000)

    if not result:
        return None

    # RapidOCR returns list of [box, text, confidence]
    texts = [item[1] for item in result if item and len(item) >= 2]
    full_text = " ".join(texts)

    m = re.search(r"(\d{1,2})\s*发言中", full_text)
    if m:
        num = m.group(1).lstrip("0") or "0"
        speaker = num.zfill(2)
        print(f"[OCR] Speaker: {speaker} ({elapsed_ms}ms)", flush=True)
        return speaker

    return None


def _load_templates():
    global _templates
    if _templates is None:
        _templates = {}
        template_files = ["01", "02", "06", "10", "11", "12", "13"]
        for digit in template_files:
            template_file = f"{template_path}/{digit}.png"
            if os.path.exists(template_file):
                _templates[digit] = cv2.imread(template_file, 0)
    return _templates


def _recognize_image_template(card_top: np.ndarray) -> Optional[str]:
    templates = _load_templates()
    if not templates:
        return None

    best_digit = None
    best_score = 0.0
    threshold = 0.8

    for digit, template in templates.items():
        if template is None:
            continue
        res = cv2.matchTemplate(card_top, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_digit = digit

    if best_score >= threshold and best_digit is not None:
        return best_digit
    return None


def _recognize_image_template_multiscale(img_gray: np.ndarray) -> Optional[str]:
    """
    Fallback recognizer: search templates on the whole frame with a small scale pyramid.
    This is more robust when ROI extraction fails due to UI/layout changes.
    """
    templates = _load_templates()
    if not templates:
        return None

    best_digit = None
    best_score = 0.0
    # Whole-frame search is noisier; keep a higher threshold.
    threshold = 0.88
    scales = (0.80, 0.90, 1.00, 1.10, 1.20)

    for digit, template in templates.items():
        if template is None:
            continue
        th, tw = template.shape[:2]
        for s in scales:
            nh, nw = max(8, int(th * s)), max(8, int(tw * s))
            if nh >= img_gray.shape[0] or nw >= img_gray.shape[1]:
                continue
            tmpl = cv2.resize(template, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
            try:
                res = cv2.matchTemplate(img_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = float(max_val)
                best_digit = digit

    if best_score >= threshold and best_digit is not None:
        return best_digit
    return None


def extract_player_num_from_array(img) -> Optional[str]:
    if img is None or img.size == 0:
        return None

    # ── Primary: OCR the "X发言中" text at the bottom-center of the frame ──
    img_bgr = img if (len(img.shape) == 3 and img.shape[2] == 3) else (
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    )
    digit = _ocr_extract_speaker(img_bgr)
    if digit is not None:
        return digit

    # ── Fallback 1: ROI连通域 + 模板匹配 ──
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    _, binary_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_h, t_w = img_gray.shape
    total_square = t_w * t_h

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_otsu)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        square = w * h
        white_ratio = area / square if square > 0 else 0
        if total_square * 0.01 < square < total_square * 0.02 and white_ratio > 0.7 and 2.0 < w / h < 3.0:
            card_top = binary_otsu[y : int(y + h * 0.3), x : int(x + w * 0.15)]
            result = _recognize_image_template(card_top)
            if result is not None:
                return result

    # ── Fallback 2: 整帧多尺度模板匹配 ──
    digit = _recognize_image_template_multiscale(binary_otsu)
    if digit is not None:
        return digit

    return None


_ROSTER_SKIP = {
    "发言中", "秒", "轮麦阶段", "总任务进度", "输入聊天消息",
    "请准备", "找到了", "轮麦", "阶段", "正程师", "工程师",
}

_CIRCLED = {
    "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5, "⑥": 6,
    "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10, "⑪": 11, "⑫": 12,
    "⑬": 13, "⑭": 14, "⑮": 15, "⑯": 16,
}


def _norm_num(s: str) -> str:
    """Fix common OCR confusion for digits: O→0, l/I→1."""
    return s.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")


def _valid_roster_name(name: str) -> bool:
    if not name or len(name) < 1:
        return False
    if name in _ROSTER_SKIP:
        return False
    if re.match(r'^[\d\s.]+$', name):
        return False
    return True


def extract_player_roster(img_bgr: np.ndarray) -> list[dict]:
    """
    Extract player roster from the full game screenshot.

    1. OCR the FULL image (no cropping) — get every piece of text.
    2. Print all text + coordinates to console for debugging.
    3. Filter by x-coordinate: only items in the left ~58% belong to the player grid.
    4. Match number+name patterns (merged / separate / circled glyphs).
    5. Fallback: infer player numbers from grid position (3 cols × N rows).
    """
    ocr = _get_ocr(block=True)
    if ocr is None:
        return []

    h, w = img_bgr.shape[:2]
    max_w = 1280
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img_bgr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img = img_bgr

    ih, iw = img.shape[:2]

    try:
        result, _ = ocr(img)
    except Exception as e:
        logger.error("Roster OCR failed: %s", e)
        return []

    if not result:
        logger.warning("Roster OCR: no text detected at all")
        return []

    # ── Collect every OCR item with full position info ──
    all_items: list[dict] = []
    for entry in result:
        if not entry or len(entry) < 3:
            continue
        box, text, conf = entry[0], str(entry[1]).strip(), entry[2]
        if not text:
            continue
        cy = sum(p[1] for p in box) / 4
        cx = sum(p[0] for p in box) / 4
        left_x = min(p[0] for p in box)
        right_x = max(p[0] for p in box)
        top_y = min(p[1] for p in box)
        bot_y = max(p[1] for p in box)
        all_items.append({
            "text": text, "cx": cx, "cy": cy,
            "left_x": left_x, "right_x": right_x,
            "top_y": top_y, "bot_y": bot_y, "conf": conf,
        })

    # ── Dump ALL text to console so user can debug ──
    print(f"[OCR-Roster] Detected {len(all_items)} text items (image {iw}x{ih}):", flush=True)
    for it in all_items:
        tag = "GRID" if it["cx"] < iw * 0.58 else "CHAT"
        print(
            f"  [{tag}] x={it['left_x']:.0f}~{it['right_x']:.0f}  "
            f"y={it['top_y']:.0f}~{it['bot_y']:.0f}  "
            f"'{it['text']}'  conf={it['conf']:.2f}",
            flush=True,
        )

    # ── Keep only items whose center-x is in the left 58% (the player grid) ──
    grid_x_max = iw * 0.58
    grid_items = [it for it in all_items if it["cx"] < grid_x_max]

    roster: list[dict] = []
    used: set[int] = set()

    # ── Pass 1: number+name merged (e.g. "01桌子不棋邓紫") ──
    for i, it in enumerate(grid_items):
        norm = _norm_num(it["text"])
        # circled glyph prefix
        first = norm[0] if norm else ""
        if first in _CIRCLED:
            num = _CIRCLED[first]
            name = norm[1:].strip()
        else:
            m = re.match(r'^[（(®©]?(\d{1,2})[)）]?\s*(.+)$', norm)
            if not m:
                continue
            num = int(m.group(1))
            name = m.group(2).strip()
        if not (1 <= num <= 16) or not _valid_roster_name(name):
            continue
        roster.append({"number": str(num).zfill(2), "name": name})
        used.add(i)

    # ── Pass 2: standalone number + nearest name to its right ──
    found_nums = {r["number"] for r in roster}
    for i, it in enumerate(grid_items):
        if i in used:
            continue
        norm = _norm_num(it["text"])
        m = re.match(r'^[（(®©]?(\d{1,2})[)）]?$', norm)
        if not m:
            # also accept circled glyph alone
            if len(norm) == 1 and norm in _CIRCLED:
                num = _CIRCLED[norm]
            else:
                continue
        else:
            num = int(m.group(1))
        num_str = str(num).zfill(2)
        if not (1 <= num <= 16) or num_str in found_nums:
            continue

        best_name = None
        best_dx = float("inf")
        best_j = -1
        for j, other in enumerate(grid_items):
            if j in used or j == i:
                continue
            if re.match(r'^[\d（()）®©]+$', _norm_num(other["text"])):
                continue
            dy = abs(other["cy"] - it["cy"])
            dx = other["left_x"] - it["right_x"]
            if dy < 35 and -15 < dx < 300 and abs(dx) < best_dx:
                cand = other["text"].strip()
                if _valid_roster_name(cand):
                    best_dx = abs(dx)
                    best_name = cand
                    best_j = j
        if best_name:
            roster.append({"number": num_str, "name": best_name})
            used.add(i)
            used.add(best_j)
            found_nums.add(num_str)

    # ── Pass 3 (fallback): infer numbers from grid position ──
    # The player grid is 3 columns × N rows, numbered sequentially L→R, T→B.
    # Collect ALL name-like items from the grid area (including already-matched
    # ones) to establish row/column structure, then fill gaps.
    found_nums = {r["number"] for r in roster}
    matched_names = {r["name"] for r in roster}

    name_positions: list[dict] = []
    for i, it in enumerate(grid_items):
        text = it["text"].strip()
        norm = _norm_num(text)
        if re.match(r'^[\d（()）®©]+$', norm):
            continue
        if not _valid_roster_name(text):
            continue
        # For merged "01name" items, extract just the name for comparison
        raw_name = text
        m = re.match(r'^[（(®©]?\d{1,2}[)）]?\s*(.+)$', norm)
        if m:
            raw_name = m.group(1).strip()
        already = raw_name in matched_names
        name_positions.append({
            "name": raw_name, "cx": it["cx"], "cy": it["cy"],
            "already_matched": already, "idx": i,
        })

    if name_positions:
        name_positions.sort(key=lambda x: (x["cy"], x["cx"]))
        # Cluster into rows: items within 8% of image height → same row
        row_thresh = ih * 0.08
        rows: list[list[dict]] = []
        cur_row = [name_positions[0]]
        for np_ in name_positions[1:]:
            if abs(np_["cy"] - cur_row[0]["cy"]) < row_thresh:
                cur_row.append(np_)
            else:
                rows.append(cur_row)
                cur_row = [np_]
        rows.append(cur_row)

        for row_idx, row in enumerate(rows):
            row.sort(key=lambda x: x["cx"])
            for col_idx, item in enumerate(row):
                if item["already_matched"]:
                    continue
                expected_num = row_idx * 3 + col_idx + 1
                num_str = str(expected_num).zfill(2)
                if 1 <= expected_num <= 16 and num_str not in found_nums:
                    roster.append({"number": num_str, "name": item["name"]})
                    found_nums.add(num_str)

    # ── Sort and deduplicate ──
    roster.sort(key=lambda x: x["number"])
    seen: set[str] = set()
    deduped = []
    for r in roster:
        if r["number"] not in seen:
            seen.add(r["number"])
            deduped.append(r)

    print(f"[OCR-Roster] Final roster ({len(deduped)} players):", flush=True)
    for r in deduped:
        print(f"  {r['number']} → {r['name']}", flush=True)
    logger.info("Roster OCR: found %d players: %s", len(deduped), deduped)
    return deduped


class SpeakerDigitMonitor:
    def __init__(self, callback=None, interval=0.5):
        self.callback = callback
        self.interval = interval
        self.current_digit = None
        self.is_running = False
        self.monitor_thread = None
        self._lock = threading.Lock()

    def start(self, capture_func):
        self.is_running = True
        self.capture_func = capture_func
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self):
        frame_count = 0
        while self.is_running:
            loop_start = time.time()
            try:
                img = self.capture_func()
                if img is not None:
                    digit = extract_player_num_from_array(img)
                    frame_count += 1
                    if frame_count == 1:
                        print(f"[MONITOR] First frame captured OK ({img.shape}), detected={digit}", flush=True)
                    elif frame_count % 40 == 0:
                        ocr_status = "ready" if _ocr_engine is not None else "loading..."
                        print(f"[MONITOR] Frame #{frame_count}, detected={digit}, ocr={ocr_status}", flush=True)
                    with self._lock:
                        old_digit = self.current_digit
                        if digit != old_digit and digit is not None:
                            self.current_digit = digit
                            logger.info("Speaker change: %s -> %s", old_digit, digit)
                            if self.callback:
                                self.callback(digit, old_digit)
                elif frame_count == 0:
                    logger.warning("Screen capture returned None on first frame — check window handle")
            except Exception as e:
                logger.error("Monitor loop error: %s", e, exc_info=True)

            elapsed = time.time() - loop_start
            time.sleep(max(0, self.interval - elapsed))

    def get_current_digit(self):
        with self._lock:
            return self.current_digit


class ChatBoxTextMonitor:
    def __init__(self, callback=None, interval=1.0):
        self.callback = callback
        self.interval = interval
        self.current_text = ""
        self.is_running = False
        self.monitor_thread = None
        self._lock = threading.Lock()

    def start(self, capture_func):
        self.is_running = True
        self.capture_func = capture_func
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self):
        while self.is_running:
            loop_start = time.time()
            try:
                img = self.capture_func()
                if img is not None:
                    text = extract_chat_box_text(img).strip()
                    with self._lock:
                        old_text = self.current_text
                        if text and text != old_text:
                            self.current_text = text
                            if self.callback:
                                self.callback(text, old_text)
            except Exception as e:
                logger.error("Chat OCR monitor loop error: %s", e, exc_info=True)
            elapsed = time.time() - loop_start
            time.sleep(max(0, self.interval - elapsed))

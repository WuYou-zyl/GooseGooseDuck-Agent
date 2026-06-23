import numpy as np
import cv2
import win32gui
import win32ui
import win32con
from win32api import GetSystemMetrics
from ctypes import windll, wintypes

from .extract_speaker_num import ChatBoxTextMonitor, SpeakerDigitMonitor, extract_player_num_from_array

user32 = windll.user32
PrintWindow = user32.PrintWindow
PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
PrintWindow.restype = wintypes.BOOL


class ScreenCapture:
    def __init__(self, hwnd=None):
        self.hwnd = hwnd
        self._dc = None
        self._cdc = None
        self._mem_dc = None
        self._bitmap = None
        self._width = 0
        self._height = 0
        self._init_capture()

    def _init_capture(self):
        if self.hwnd:
            title = win32gui.GetWindowText(self.hwnd)
            window_rect = win32gui.GetWindowRect(self.hwnd)
            print(f"[CAPTURE] hwnd={self.hwnd}, title={title!r}", flush=True)
            print(f"[CAPTURE] window_rect={window_rect}", flush=True)

            # Restore minimized windows so we can capture them
            if win32gui.IsIconic(self.hwnd):
                print("[CAPTURE] Window is minimized, restoring ...", flush=True)
                win32gui.ShowWindow(self.hwnd, 9)  # SW_RESTORE
                import time
                time.sleep(0.5)
                window_rect = win32gui.GetWindowRect(self.hwnd)

            client_rect = win32gui.GetClientRect(self.hwnd)
            left, top, right, bottom = client_rect
            self._width = right - left
            self._height = bottom - top
            print(f"[CAPTURE] client size: {self._width}x{self._height}", flush=True)

            if self._width < 200 or self._height < 200:
                wl, wt, wr, wb = window_rect
                self._width = max(wr - wl, 800)
                self._height = max(wb - wt, 600)
                print(f"[CAPTURE] Client too small, using window rect: {self._width}x{self._height}", flush=True)

            self._dc = win32gui.GetWindowDC(self.hwnd)
        else:
            self._dc = win32gui.GetDC(0)
            self._width = GetSystemMetrics(0)
            self._height = GetSystemMetrics(1)

        self._cdc = win32ui.CreateDCFromHandle(self._dc)
        self._mem_dc = self._cdc.CreateCompatibleDC()
        self._bitmap = win32ui.CreateBitmap()
        self._bitmap.CreateCompatibleBitmap(self._cdc, self._width, self._height)
        self._mem_dc.SelectObject(self._bitmap)

    def capture(self, use_fast_mode=True):
        try:
            if self.hwnd:
                # PW_RENDERFULLCONTENT=2: captures GPU-rendered content (Edge, games, etc.)
                result = PrintWindow(int(self.hwnd), int(self._mem_dc.GetSafeHdc()), 2)
                if not result:
                    # Fallback to BitBlt
                    self._mem_dc.BitBlt((0, 0), (self._width, self._height), self._cdc, (0, 0), win32con.SRCCOPY)
            else:
                self._mem_dc.BitBlt((0, 0), (self._width, self._height), self._cdc, (0, 0), win32con.SRCCOPY)

            bmpinfo = self._bitmap.GetInfo()
            bmpstr = self._bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8)

            if bmpinfo["bmBitsPixel"] == 32:
                img.shape = (self._height, self._width, 4)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif bmpinfo["bmBitsPixel"] == 24:
                img.shape = (self._height, self._width, 3)
            else:
                return None
            return img
        except Exception:
            return None

    def release(self):
        if self._bitmap:
            win32gui.DeleteObject(self._bitmap.GetHandle())
            self._bitmap = None
        if self._mem_dc:
            self._mem_dc.DeleteDC()
            self._mem_dc = None
        if self._cdc:
            self._cdc.DeleteDC()
            self._cdc = None
        if self._dc:
            if self.hwnd:
                win32gui.ReleaseDC(self.hwnd, self._dc)
            else:
                win32gui.ReleaseDC(0, self._dc)
            self._dc = None


class WindowScreenMonitor:
    def __init__(self, hwnd, on_digit_change=None, on_chat_ocr_change=None, interval=0.5):
        self.hwnd = hwnd
        self.on_digit_change = on_digit_change
        self.on_chat_ocr_change = on_chat_ocr_change
        self.interval = interval
        self.screen_capture = None
        self.digit_monitor = None
        self.chat_monitor = None
        self.current_digit = None
        self.crop_region = None  # (x, y, w, h) in pixels, or None for full window

    def set_crop_region(self, x: int, y: int, w: int, h: int):
        """Set a sub-region of the window to monitor. Coordinates in pixels."""
        self.crop_region = (x, y, w, h)
        print(f"[CAPTURE] Crop region set: x={x}, y={y}, w={w}, h={h}", flush=True)

    def clear_crop_region(self):
        self.crop_region = None
        print("[CAPTURE] Crop region cleared, using full window", flush=True)

    def _capture_func(self):
        if self.screen_capture:
            img = self.screen_capture.capture()
            if img is not None and self.crop_region:
                x, y, w, h = self.crop_region
                ih, iw = img.shape[:2]
                x = max(0, min(x, iw - 1))
                y = max(0, min(y, ih - 1))
                w = min(w, iw - x)
                h = min(h, ih - y)
                if w > 10 and h > 10:
                    img = img[y:y+h, x:x+w]
            return img
        return None

    def _on_digit_callback(self, new_digit, old_digit):
        self.current_digit = new_digit
        if self.on_digit_change:
            self.on_digit_change(new_digit, old_digit)

    def _on_chat_ocr_callback(self, new_text, old_text):
        if self.on_chat_ocr_change:
            self.on_chat_ocr_change(new_text, old_text)

    def start(self):
        self.screen_capture = ScreenCapture(self.hwnd)
        self.digit_monitor = SpeakerDigitMonitor(callback=self._on_digit_callback, interval=self.interval)
        self.digit_monitor.start(self._capture_func)
        self.chat_monitor = ChatBoxTextMonitor(callback=self._on_chat_ocr_callback, interval=1.0)
        self.chat_monitor.start(self._capture_func)

    def stop(self):
        if self.digit_monitor:
            self.digit_monitor.stop()
        if self.chat_monitor:
            self.chat_monitor.stop()
        if self.screen_capture:
            self.screen_capture.release()

    def get_current_digit(self):
        if self.digit_monitor:
            return self.digit_monitor.get_current_digit()
        return None

    def capture_and_detect(self):
        if not self.screen_capture:
            self.screen_capture = ScreenCapture(self.hwnd)
        img = self.screen_capture.capture()
        if img is not None:
            digit = extract_player_num_from_array(img)
            return img, digit
        return None, None

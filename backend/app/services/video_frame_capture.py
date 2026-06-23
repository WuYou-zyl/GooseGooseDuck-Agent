import time
import threading
from typing import Optional, Literal
import numpy as np
import cv2
import win32gui
import win32ui
from ctypes import windll
from ppadb.client import Client as AdbClient
import subprocess
from backend.utils.logger import log
from backend.app.core.ws_manager import ws_manager
import asyncio

class VideoFrameCaptureService:
    def __init__(self, mode: Literal["adb", "window"] = "window", target: str = "", vision_service=None, loop: asyncio.AbstractEventLoop = None, fps_limit: int = 2):
        """
        :param mode: "adb" (模拟器) 或 "window" (播放器窗口)
        :param target: 如果是 window 模式，传入窗口标题关键词；如果是 adb 模式，传入 adb 连接字符串
        """
        self.mode = mode
        self.target = target
        self.is_running = False
        self.latest_frame = None
        self.thread = None
        self._config_lock = threading.RLock()
        
        # 缓存的资源
        self._adb_device = None
        self._target_hwnd = None
        
        # 性能控制,默认 1s 截取 2帧
        self.fps_limit = fps_limit
        self.interval = 1.0 / self.fps_limit

        # 注入触发视觉处理逻辑
        self.vision_service = vision_service

        # 获取主线程的事件循环引用
        if not loop:
            self.loop = asyncio.get_event_loop()
        else:
            self.loop = loop

    def _init_adb(self):
        """初始化 ADB 环境：启动 Server -> 执行 Connect -> 绑定 Device"""
        try:
            # 1. 检查并启动 ADB Server
            # check_output 会尝试执行 adb version，如果失败说明 adb 没装或没在环境变量里
            subprocess.run(["adb", "start-server"], check=True, capture_output=True)

            # 2. 尝试执行 adb connect
            if ":" in self.target: # 确保 target 是类似 127.0.0.1:16384 的格式
                log.info(f"正在尝试连接 ADB 设备: {self.target}...")
                # shell=True 在 Windows 下有助于处理复杂的路径，但通常直接传列表更安全
                result = subprocess.run(["adb", "connect", self.target], capture_output=True, text=True)
                log.info(f"ADB Connect 响应: {result.stdout.strip()}")

            # 3. 实例化 ppadb Client
            client = AdbClient(host="127.0.0.1", port=5037)
            
            # 4. 绑定设备
            devices = client.devices()
            if not devices:
                log.error(f"ADB: 无法找到设备。请检查模拟器是否启动或端口 {self.target} 是否正确。")
                return False
            
            # 寻找匹配的设备
            for d in devices:
                if self.target in d.serial:
                    self._adb_device = d
                    break
            
            # 如果没找到完全匹配的且设备列表不为空，降级取第一个
            if not self._adb_device and devices:
                self._adb_device = devices[0]
                
            log.info(f"✅ ADB 捕获环境初始化成功: {self._adb_device.serial}")
            return True

        except FileNotFoundError:
            log.error("系统未找到 adb 执行文件，请检查是否已安装 ADB 并在环境变量中配置了 PATH。")
            return False
        except Exception as e:
            log.error(f"ADB 自动化初始化失败: {e}")
            return False

    def _init_window(self):
        """查找窗口句柄并缓存"""
        hwnd = 0
        def callback(current_hwnd, _):
            nonlocal hwnd
            title = win32gui.GetWindowText(current_hwnd)
            if self.target.lower() in title.lower():
                hwnd = current_hwnd
                return False
            return True
        
        win32gui.EnumWindows(callback, None)
        if hwnd:
            self._target_hwnd = hwnd
            log.info(f"✅ 窗口捕获已准备就绪: {win32gui.GetWindowText(hwnd)} (HWND: {hwnd})")
            return True
        else:
            log.warning(f"未找到目标窗口: {self.target}")
            return False

    def _capture_loop(self):
        """核心捕获循环"""
        log.info(f"开始启动 {self.mode} 捕获线程...")
        
        while self.is_running:
            start_time = time.perf_counter()
            frame = None

            try:
                with self._config_lock:
                    mode = self.mode

                if mode == "adb":
                    if not self._adb_device:
                        self._init_adb()
                    else:
                        screenshot = self._adb_device.screencap()
                        if screenshot:
                            # 转换为 numpy -> 图像解码
                            img_array = np.frombuffer(screenshot, dtype='uint8')
                            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                elif mode == "window":
                    if not self._target_hwnd or not win32gui.IsWindow(self._target_hwnd):
                        self._init_window()
                    else:
                        frame = self._capture_window_logic()

                if frame is not None:
                    self.latest_frame = frame
                    # 在后端闭环处理
                    if self.vision_service:
                        speaker_info = self.vision_service.process_frame(frame)
                        # 注意这个方法是 async 的，同步中调用异步需要使用 run_coroutine_threadsafe
                        asyncio.run_coroutine_threadsafe(
                            ws_manager.broadcast(speaker_info),
                            self.loop    
                        )
                        
            except Exception as e:
                log.error(f"捕获循环异常: {e}")
                time.sleep(1) # 异常后稍作等待

            # 控制 FPS
            elapsed = time.perf_counter() - start_time
            with self._config_lock:
                interval = self.interval
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    def _capture_window_logic(self):
        """Win32 窗口截图逻辑"""
        hwnd = self._target_hwnd
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w, h = right - left, bot - top
        
        if w <= 0 or h <= 0: return None

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        # 执行截图
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
        
        if result == 1:
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (h, w, 4)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            frame = None

        # 释放资源
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        return frame

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            log.info("视频帧捕获服务已启动")

    def update_config(self, mode: str, target: str, fps_limit: int):
        fps_limit = max(1, int(fps_limit))
        with self._config_lock:
            source_changed = self.mode != mode or self.target != target
            self.mode = mode
            self.target = target
            self.fps_limit = fps_limit
            self.interval = 1.0 / fps_limit

            if source_changed:
                self._adb_device = None
                self._target_hwnd = None
                self.latest_frame = None

        log.info(f"视频采集配置已热更新: mode={mode}, target={target}, fps_limit={fps_limit}")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        log.info("视频帧捕获服务已停止")

    def get_latest_frame(self):
        """供主程序调用的接口"""
        return self.latest_frame

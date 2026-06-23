import cv2
import numpy as np
import threading
import queue
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from backend.utils.logger import log
from backend.utils.pathtool import get_abs_path
from backend.app.core.ggd_coordinator import GGDCoordinator
# from paddleocr import PaddleOCR
from rapidocr import RapidOCR

_GGD_GRID_ROWS = 3
_GGD_GRID_COLS = 5
_GGD_ROW_RATIO = 1.06
_GGD_SEAT_COUNT = _GGD_GRID_COLS * _GGD_GRID_ROWS

class GGDVisionService:
    def __init__(self, config_path="roi_config.json"):
        log.info("初始化 GGDVisionService（带独立 OCR 线程）...")
        self.config_path = config_path
        self.ocr_config_path = get_abs_path("rapidocr.yaml")
        
        # 共享状态
        self.seat_rois = []       # 15个座位的坐标 [(x, y, w, h), ...]
        self.speaker_ui_roi = None # 用户框选的发言提示区坐标 (x, y, w, h)
        self.seat_names = {}      # 座位号 -> 玩家ID {1: "小明", ...}
        self.current_ui_text = "" # 当前实时识别到的 UI 文本

        # 默认参与游戏人数
        self.seat_num = 13
        self._seat_lock = threading.RLock()

        # 协调器实例，需要在 main.py 中注入
        self.coordinator : GGDCoordinator = None

        try:
            # self.ocr_engine = PaddleOCR(
            #     use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
            #     use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
            #     use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
            #     ocr_version="PP-OCRv5",
            #     device="cpu",
            # )
            
            self.ocr_engine = RapidOCR(
                config_path=str(self.ocr_config_path),
            )

            
            # 线程通信：队列长度设为 1，确保 OCR 线程只处理最新的帧
            self.frame_queue = queue.Queue(maxsize=1)
            self.current_rois = None  # 线程共享资源：当前的座位网格
            self.is_running = True

            # 加载标定数据
            self._load_roi_config()
            
            # 启动独立扫描线程
            self.scan_thread = threading.Thread(target=self._ocr_scan_loop, name="OCR_Scanner", daemon=True)
            self.scan_thread.start()
            
            log.success("视觉服务与 OCR 后台线程启动成功。")
        except Exception as e:
            log.error(f"视觉服务初始化失败: {e}")
            raise e
    
    def set_seat_num(self, num: int):
        with self._seat_lock:
            self.seat_num = max(1, min(int(num), _GGD_SEAT_COUNT))
            self.seat_names = {
                seat: name
                for seat, name in self.seat_names.items()
                if seat <= self.seat_num
            }
        log.info(f"游戏人数已更新为 {self.seat_num}")

    def _load_roi_config(self):
        """从 JSON 加载标定数据并推导完整网格"""
        if not os.path.exists(self.config_path):
            log.warning(f"未找到标定文件 {self.config_path}，请先进行界面标定。")
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 获取用户标定的三个核心座位坐标 (假设编号从1开始)
            # p11: 座位1, p12: 座位2, p21: 座位4 (5x3布局下的第二行第一位)
            p11 = config['seat_1'] # (x, y, w, h)
            p12 = config['seat_2']
            p21 = config['seat_4'] 
            self.speaker_ui_roi = config.get('speaker_ui')

            dx = p12[0] - p11[0] # 水平间距
            dy = p21[1] - p11[1] # 垂直间距
            w, h = p11[2], p11[3]

            # 推导出 3x5 的 15 个 ROI
            self.seat_rois = []
            for row in range(5):
                for col in range(3):
                    curr_x = p11[0] + col * dx
                    curr_y = p11[1] + row * dy
                    self.seat_rois.append((curr_x, curr_y, w, h))
            
            log.success(f"标定配置加载成功，推导出 {len(self.seat_rois)} 个座位坐标。")
            return {
                "status": "success",
                "config_preview": {
                    "seats": self.seat_rois,        # 15个座位的 [x, y, w, h] 列表
                    "speaker_ui": self.speaker_ui_roi # 提示区的 [x, y, w, h]
                }
            }
        except Exception as e:
            log.error(f"加载标定配置失败: {e}")

    def update_config(self):
        """
        供外部(API)调用：当用户在前端完成标定并保存文件后，
        手动触发此函数重新读取 JSON 并更新内存中的坐标。
        """
        log.info("接收到更新指令，正在重新加载标定配置...")
        return self._load_roi_config()

    def _ocr_scan_loop(self):
        """后台线程：同时处理 ID 识别（直到完成）和 UI 提示词识别（持续）"""
        log.info("OCR 扫描循环已启动")
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # 1. 持续识别 UI 发言提示（优先级高）
                if self.speaker_ui_roi:
                    x, y, w, h = self.speaker_ui_roi
                    ui_crop = frame[y:y+h, x:x+w]
                    if ui_crop.size > 0:
                        res = self.ocr_engine(ui_crop)
                        if res:
                            self.current_ui_text = " ".join(res.txts)

                # 2. 识别玩家 ID（如果还有座位没识别出来）
                with self._seat_lock:
                    seat_num = self.seat_num

                if len(self.seat_names) < seat_num:
                    for i, (x, y, w, h) in enumerate(self.seat_rois[:seat_num], 1):
                        if i in self.seat_names: continue
                        
                        # 截取名字区域（假设名字在卡片上半部分，取 ROI 的前 30%）
                        name_crop = frame[y:y+int(h*0.3), x:x+w]
                        # cv2.imwrite(f"debug_{i}.jpg", name_crop)
                        res = self.ocr_engine(name_crop)
                        if res:
                            name = "".join(res.txts)
                            if name: 
                                self.seat_names[i] = name
                                log.info(f"检测到座位 {i}: {name}")

                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"OCR 线程异常: {e}")

    def process_frame(self, img: np.ndarray) -> Dict:
        """处理单帧图像"""
        if not self.seat_rois:
            return {"status": "waiting_calibration", "msg": "请先完成标定"}

        try:
            self.frame_queue.put_nowait(img)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(img)
        
        # 1. 检测发言状态
        scores = self._detect_speaker_status(img)
        max_score = max(scores.values()) if scores else 0.0
        speaker_id = max(scores, key=scores.get) if scores and max_score > 0.1 else None   # 默认遍历字典的 key，key=scores.get 是告诉 max 如何进行比较
        
        # 2. 识别名字
        speaker_name = "Unknown"
        if speaker_id is not None:
            speaker_name = self.seat_names.get(speaker_id, "Unknown")

        # 在 process_frame 内部
        res = {
            "active_seat": speaker_id, 
            "ui_text": self.current_ui_text,
        }
        # 如果外面传入了 coordinator 实例
        if self.coordinator:
            self.coordinator.record_vision_state(res)
        
        return {
            "type": "processed_frame",
            "active_seat": speaker_id,
            "speaker_name": speaker_name
        }
    
    def _get_seat_rois(self, img: np.ndarray) -> Optional[List[Tuple]]:
        """已弃用：座位网格推导逻辑（耗时操作）[cite: 3]"""
        
        def preprocess_img(img_bgr: np.ndarray) -> np.ndarray:
            '''
            图像预处理，包括灰度化、自适应直方图均衡化、二值化、形态学开运算
            使得座位卡片区域更加清晰
            '''
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray)
            _, binary = cv2.threshold(cl1, 30, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 25))
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        def ggd_find_anchors(
            processed_img: np.ndarray,
            original_shape: Tuple[int, ...],
            ) -> List[Dict[str, Any]]:
            '''
            寻找座位卡片底部锚点（14/15 号卡片）
            '''
            h_img, w_img = original_shape[:2]
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
            candidates: List[Dict[str, Any]] = []
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                aspect_ratio = w / float(h) if h else 0.0
                is_correct_area = 0.01 * w_img * h_img < area < 0.1 * w_img * h_img
                is_correct_ratio = 1.5 < aspect_ratio < 3.5
                is_left_side = x < w_img * 0.65
                if is_correct_area and is_correct_ratio and is_left_side:
                    candidates.append({"rect": (x, y, w, h), "x": x, "y": y, "area": int(area)})

            if len(candidates) < 2:
                return []

            candidates.sort(key=lambda c: c["y"], reverse=True)
            bottom_row = candidates[:3]

            if len(bottom_row) == 2:
                bottom_row.sort(key=lambda c: c["x"])
                return bottom_row

            min_diff = float("inf")
            best_pair: List[Dict[str, Any]] = []
            for i in range(len(bottom_row)):
                for j in range(i + 1, len(bottom_row)):
                    diff = abs(bottom_row[i]["area"] - bottom_row[j]["area"])
                    if diff < min_diff:
                        min_diff = diff
                        best_pair = [bottom_row[i], bottom_row[j]]

            best_pair.sort(key=lambda c: c["x"])
            return best_pair
        
        def ggd_derive_grid(img_bgr: np.ndarray, anchors: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
            '''
            从底部锚点（14/15 号卡片）推导出 3x5 网格
            '''
            if len(anchors) < 2:
                return []

            x14, y14, w14, h14 = anchors[0]["rect"]
            x15, y15, w15, h15 = anchors[1]["rect"]
            h_img, w_img = img_bgr.shape[:2]
            avg_w = (w14 + w15) / 2
            avg_h = (h14 + h15) / 2
            delta_x = x15 - x14
            delta_y = avg_h * _GGD_ROW_RATIO

            rois: List[Tuple[int, int, int, int]] = []
            for row in range(_GGD_GRID_ROWS):
                for col in range(_GGD_GRID_COLS):
                    curr_x = int(x14 + (col - 1) * delta_x)
                    curr_y = int(y14 + (row - 4) * delta_y)
                    curr_w, curr_h = int(avg_w * 1), int(avg_h)
                    curr_x = max(0, min(w_img - curr_w, curr_x))
                    curr_y = max(0, min(h_img - curr_h, curr_y))
                    rois.append((curr_x, curr_y, curr_w, curr_h))
            return rois
    
        binary = preprocess_img(img_bgr=img)
        anchors = ggd_find_anchors(binary, img.shape)
        card_rois = ggd_derive_grid(img_bgr=img, anchors=anchors)
        if len(card_rois) != _GGD_GRID_ROWS * _GGD_GRID_COLS:
            return None
        return card_rois

    def _detect_speaker_status(self, img: np.ndarray) -> Dict[int, float]:
        """
        HSV 颜色空间检测发言黄色边框
        优化方案：仅计算 ROI 边缘 5% 范围内的黄色像素占比
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 标准 GGD 黄色边框范围
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        scores = {}
        for i, (x, y, w, h) in enumerate(self.seat_rois, 1):
            if y < 0 or x < 0: continue
            
            # 1. 截取原始座位卡片区域
            roi = hsv[y:y+h, x:x+w]
            if roi.size == 0: continue
            
            # 2. 定义边框厚度 (1/20)，考虑到偏移，增加至 1/6 高度进行容错
            thickness = max(1, int(h / 6))
            
            # 3. 构造掩码只保留边缘
            # 创建一个和 ROI 一样大的全黑遮罩
            border_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            # 将中间部分设为 0 (黑色)，四周设为 255 (白色)
            border_mask[:thickness, :] = 255             # 上边框
            border_mask[-thickness:, :] = 255            # 下边框
            border_mask[:, :thickness] = 255             # 左边框
            border_mask[:, -thickness:] = 255            # 右边框
            
            # 4. 执行颜色检测
            color_mask = cv2.inRange(roi, lower_yellow, upper_yellow)
            
            # 5. 合并掩码：只有在“边缘区域”且“颜色匹配”的像素才计数
            final_mask = cv2.bitwise_and(color_mask, border_mask)
            
            # 6. 计算得分：(匹配的边框像素) / (总边框区域像素)
            border_area = np.sum(border_mask > 0)
            scores[i] = float(np.sum(final_mask > 0) / border_area) if border_area > 0 else 0.0
            
        return scores
    
    def stop(self):
        self.is_running = False
        log.info("停止视觉服务线程...")

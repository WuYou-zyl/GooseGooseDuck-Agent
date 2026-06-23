"""Ingestion UI + monitoring API (A-side) and WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import traceback
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend import app_state
from backend.agents.ingestion import IngestionAgent
from backend.schemas.contract import DecisionOutput, IngestionOutput, iso_now
from backend.schemas.graph_state import MemoryDecisionState, PlayerRosterEntry
from backend.utils.color_roster_defaults import color_for_seat
from backend.services.tingwu_phrase_manager import TingwuPhraseManager
from backend.utils.session_checkpoint import ensure_session_state

logger = logging.getLogger("ggd-a")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FRONTEND_INDEX = os.path.join(_REPO_ROOT, "frontend", "index.html")


def _load_input_processing_config():
    from backend.utils.config_loader import load_config

    return load_config()


async def memory_graph_ingestion_consumer(ingestion: IngestionOutput) -> None:
    """IngestionAgent.consumer：与 POST /api/v1/ingestion 一致，写入 MemoryGraph。"""
    g = app_state.graph
    if g is None:
        logger.warning("Memory graph not ready; dropping ingestion")
        return
    await g.ainvoke(ingestion, ingestion.session_id)


def _forward_ingestion_via_emit(agent: IngestionAgent, out: IngestionOutput) -> None:
    """ASR 线程：经 IngestionAgent.emit（走 consumer）写入图，与 ingest_speech_frames 共用。"""
    loop = app_state.main_loop
    if loop is None:
        logger.warning("main loop not ready; dropping ingestion")
        return

    async def _run() -> None:
        await agent.emit(out)

    fut = asyncio.run_coroutine_threadsafe(_run(), loop)
    try:
        fut.result(timeout=120)
    except Exception as e:
        logger.warning(
            "Failed to forward ingestion to memory graph: %s\n%s",
            e,
            traceback.format_exc(),
        )


async def _merge_gemini_roster_into_state(session_id: str, roster: list[dict]) -> None:
    """将 Gemini 名单按 seat_number 合并进 graph_state.situation_sketch.player_roster（同座覆盖）。"""
    g = app_state.graph
    if g is None or not roster:
        return
    config = {"configurable": {"thread_id": session_id}}
    snap = await g.graph.aget_state(config)
    raw = snap.values
    if not raw:
        return
    state = MemoryDecisionState.model_validate(raw)
    by_seat: dict[int, PlayerRosterEntry] = {
        e.seat_number: e for e in state.situation_sketch.player_roster
    }
    for r in roster:
        try:
            num = int(str(r.get("number", "0")).strip())
        except ValueError:
            continue
        if not (1 <= num <= 16):
            continue
        name = (r.get("name") or "").strip() or str(num)
        by_seat[num] = PlayerRosterEntry(
            player_id=name,
            seat_number=num,
            color=color_for_seat(num),
            status="存活",
        )
    merged = sorted(by_seat.values(), key=lambda x: x.seat_number)
    new_sketch = state.situation_sketch.model_copy(update={"player_roster": merged})
    payload = {"situation_sketch": new_sketch}
    update_fn = getattr(g.graph, "aupdate_state", None)
    if update_fn is not None:
        await update_fn(config, payload, as_node="memory_draft")
    else:
        g.graph.update_state(config, payload, as_node="memory_draft")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]):
        disconnected: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)


manager = ConnectionManager()
event_queue: queue.Queue[dict[str, Any]] = queue.Queue()


async def process_events():
    while True:
        try:
            while not event_queue.empty():
                event = event_queue.get_nowait()
                await manager.broadcast(event)
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


router = APIRouter(tags=["ingestion"])


class StartMonitoringRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="本局游戏 ID（LangGraph thread_id）")


class NewGameRequest(BaseModel):
    previous_session_id: Optional[str] = Field(
        None,
        description="上一局的 session_id，若提供则先删除该 thread 的 state",
    )


class StatusResponse(BaseModel):
    status: str
    is_running: bool
    current_speaker: Optional[str] = None
    window_title: Optional[str] = None
    session_id: Optional[str] = None
    meeting_id: Optional[str] = None


class MonitorController:
    def __init__(self):
        self.hwnd: Optional[int] = None
        self.window_title: Optional[str] = None
        self.is_running: bool = False
        self._lock = threading.Lock()

        self._screen_monitor = None
        self._audio_analyzer = None
        self._current_speaker: Optional[str] = None
        self._agent: Optional[IngestionAgent] = None
        self._meeting_id: Optional[str] = None
        self.roster: list[dict] = []
        self._backend_mode: str = "legacy"
        self._optimized_vision_service = None
        self._optimized_frame_capture_service = None
        self._optimized_audio_service = None
        self._optimized_coordinator = None
        self._optimized_audio_thread: Optional[threading.Thread] = None
        self._optimized_sent_content: dict[str, str] = {}
        self._partial_text_by_speaker: dict[str, str] = {}
        self._latest_chat_ocr_text: str = ""

    def _dynamic_name_hotwords(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in self.roster:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _on_digit_change(self, new_digit, old_digit):
        """
        当屏幕监控识别到说话人数字编号（如玩家号码）变化时的回调函数，更新当前说话人。
        并通过事件队列广播“speaker_change”事件。它还会通知音频分析器设置新的说话人。
        """
        print(f"[SPEAKER] Change: {old_digit} -> {new_digit}", flush=True)
        self._current_speaker = new_digit
        event_queue.put(
            {
                "type": "speaker_change",
                "data": {"new_speaker": new_digit, "old_speaker": old_digit, "timestamp": iso_now()},
            }
        )
        if self._audio_analyzer:
            self._audio_analyzer.set_speaker(new_digit, round_num=1)

    def _on_new_record(self, record: dict[str, Any]):
        """
        当音频分析器（ASR）生成新的语音转文字记录时的回调函数。
        它将语音文本封装为 IngestionOutput数据结构，通过事件队列广播“ingestion”事件，
        并启动一个新线程调用 _forward_ingestion_via_emit 将数据发送到 IngestionAgent进行处理和写入内存图。
        """
        print(f"[ASR] New record: speaker={record.get('speaker')}, text={record.get('text', '')!r}", flush=True)
        if not self._agent:
            return
        speaker_id = record.get("speaker") or self._current_speaker
        emotion = record.get("emotion")
        text = record.get("text", "")
        mid = self._meeting_id
        meta = {
            **({"speaker_id": speaker_id} if speaker_id else {}),
            **({"emotion_summary": str(emotion)} if emotion else {}),
            "source": "legacy_audio_analyzer",
        }
        original_text = str(record.get("original_text") or "").strip()
        if original_text and original_text != text:
            meta["original_text"] = original_text
        if record.get("text_normalized"):
            meta["text_normalized"] = True
        rules = record.get("normalizer_rules")
        if isinstance(rules, list) and rules:
            meta["normalizer_rules"] = rules
        if mid:
            meta["meeting_id"] = mid
        out = IngestionOutput(
            type="speech",
            content=text,
            metadata=meta,
            timestamp=iso_now(),
            session_id=self._agent.session_id,
            meeting_id=mid,
            sequence_id=record.get("id") or None,
        )
        event_queue.put({"type": "ingestion", "data": out.model_dump()})
        threading.Thread(target=_forward_ingestion_via_emit, args=(self._agent, out), daemon=True).start()

    def _on_partial_record(self, record: dict[str, Any]):
        speaker_id = str(record.get("speaker") or self._current_speaker or "??")
        text = str(record.get("text") or "").strip()
        if not text:
            return
        if self._partial_text_by_speaker.get(speaker_id) == text:
            return
        self._partial_text_by_speaker[speaker_id] = text
        event_queue.put(
            {
                "type": "partial_ingestion",
                "data": {
                    "speaker_id": speaker_id,
                    "text": text,
                    "original_text": record.get("original_text") or text,
                    "text_normalized": bool(record.get("text_normalized")),
                    "timestamp": record.get("timestamp") or iso_now(),
                },
            }
        )

    def _on_chat_ocr_change(self, new_text: str, old_text: str):
        text = str(new_text or "").strip()
        if not text or text == self._latest_chat_ocr_text:
            return
        self._latest_chat_ocr_text = text
        event_queue.put(
            {
                "type": "chat_ocr",
                "data": {
                    "text": text,
                    "old_text": str(old_text or "").strip(),
                    "timestamp": iso_now(),
                },
            }
        )

    async def _on_optimized_speech(self, payload: dict[str, Any]) -> None:
        if not self._agent:
            return
        full_text = str(payload.get("content") or "").strip()
        if not full_text:
            return

        record_id = str(payload.get("id") or uuid.uuid4().hex)
        previous = self._optimized_sent_content.get(record_id, "")
        if payload.get("type") == "update" and previous:
            if full_text == previous:
                return
            text = full_text[len(previous):].strip() if full_text.startswith(previous) else full_text
        else:
            text = full_text
        if not text:
            return
        self._optimized_sent_content[record_id] = full_text

        speaker_id = payload.get("seat_id")
        speaker = str(speaker_id) if speaker_id is not None else None
        if speaker != self._current_speaker:
            old = self._current_speaker
            self._current_speaker = speaker
            event_queue.put(
                {
                    "type": "speaker_change",
                    "data": {
                        "new_speaker": speaker,
                        "old_speaker": old,
                        "timestamp": iso_now(),
                    },
                }
            )

        mid = self._meeting_id
        metadata = {
            "source": "input_processing",
            "fusion_method": payload.get("method"),
        }
        if speaker:
            metadata["speaker_id"] = speaker
        if payload.get("name"):
            metadata["speaker_name"] = payload.get("name")
        if mid:
            metadata["meeting_id"] = mid

        try:
            sequence_id = int(payload.get("id")) if payload.get("id") is not None else None
        except (TypeError, ValueError):
            sequence_id = None

        out = IngestionOutput(
            type="speech",
            content=text,
            metadata=metadata,
            timestamp=iso_now(),
            session_id=self._agent.session_id,
            meeting_id=mid,
            sequence_id=sequence_id,
        )
        event_queue.put({"type": "ingestion", "data": out.model_dump()})
        await self._agent.emit(out)

    def init(self):
        return True

    def input_processing_enabled(self) -> bool:
        try:
            return bool(_load_input_processing_config().enabled)
        except Exception as e:
            logger.warning("Failed to load input-processing config: %s", e)
            return False

    def attach_to_app_state(self, state) -> None:
        state.vision_service = self._optimized_vision_service
        state.frame_capture_service = self._optimized_frame_capture_service
        state.audio_service = self._optimized_audio_service
        state.ggd_coordinator = self._optimized_coordinator

    async def start_optimized(self, session_id: str, meeting_id: str, state) -> bool:
        from backend.app.core.ggd_coordinator import GGDCoordinator
        from backend.app.services.audio_service.audio_capture import AudioCaptureService
        from backend.app.services.video_frame_capture import VideoFrameCaptureService
        from backend.app.services.vision_ocr import GGDVisionService

        cfg = _load_input_processing_config()
        with self._lock:
            if self.is_running:
                return True
            self._backend_mode = "optimized"
            self._meeting_id = meeting_id
            self._optimized_sent_content = {}
            self._partial_text_by_speaker = {}
            self._latest_chat_ocr_text = ""
            self._agent = IngestionAgent(
                session_id=session_id,
                consumer=memory_graph_ingestion_consumer,
            )

        loop = asyncio.get_running_loop()
        vision_service = GGDVisionService()
        vision_service.set_seat_num(cfg.game_setting.seat_num)
        coordinator = GGDCoordinator(
            vision_service,
            loop=loop,
            on_speech_result=self._on_optimized_speech,
        )
        vision_service.coordinator = coordinator

        frame_capture_service = VideoFrameCaptureService(
            cfg.vision.mode,
            cfg.vision.target,
            vision_service=vision_service,
            loop=loop,
            fps_limit=cfg.vision.fps_limit,
        )
        audio_service = AudioCaptureService(coordinator)
        audio_thread = threading.Thread(
            target=audio_service.run,
            name="OptimizedAudioCaptureThread",
            daemon=True,
        )

        with self._lock:
            self._optimized_vision_service = vision_service
            self._optimized_coordinator = coordinator
            self._optimized_frame_capture_service = frame_capture_service
            self._optimized_audio_service = audio_service
            self._optimized_audio_thread = audio_thread
            self.window_title = cfg.vision.target or cfg.vision.mode
            self.is_running = True
            self.attach_to_app_state(state)

        frame_capture_service.start()
        audio_thread.start()
        event_queue.put({"type": "status_change", "data": {"status": "running", "timestamp": iso_now()}})
        return True

    def select_window(self):
        """让用户选择要监控的游戏窗口，并记录窗口句柄（hwnd）和标题。"""
        try:
            from backend.legacy.window_selector import select_window  # lazy import (pywin32/tk deps)
        except Exception as e:
            raise RuntimeError(
                "窗口选择不可用：缺少依赖（通常是 pywin32/tkinter）。"
                "请安装 `pywin32` 后重试。原始错误："
                f"{e}"
            )

        hwnd, title = select_window()
        if hwnd is None:
            return None, None
        self.hwnd = int(hwnd)
        self.window_title = str(title)
        return self.hwnd, self.window_title

    def start(self, session_id: str, meeting_id: str):
        if self.input_processing_enabled():
            raise RuntimeError("input_processing is enabled; use start_optimized from the async API route")

        try:
            from backend.legacy.screen_monitor import WindowScreenMonitor  # lazy import (pywin32 deps)
            from backend.services.system_audio_realtime_service import SystemAudioRealtimeService
        except Exception as e:
            raise RuntimeError(
                "启动读入监控失败：缺少依赖。请按仓库根目录 `requirements.txt` 安装依赖。"
                f" 原始错误：{e}"
            )

        with self._lock:
            if self.is_running:
                return True
            if self.hwnd is None:
                raise RuntimeError("未选择窗口")

            # 创建 IngestionAgent 实例
            self._meeting_id = meeting_id
            self._partial_text_by_speaker = {}
            self._latest_chat_ocr_text = ""
            print(f"[INIT] Creating IngestionAgent (session={session_id}) ...", flush=True)
            self._agent = IngestionAgent(
                session_id=session_id,
                consumer=memory_graph_ingestion_consumer,
            )
            # 加载 ASR 模型
            print(f"[INIT] Initializing online ASR (Tingwu realtime) ...", flush=True)
            self._audio_analyzer = SystemAudioRealtimeService(
                on_new_record=self._on_new_record,
                auto_save=True,
                dynamic_hotwords_provider=self._dynamic_name_hotwords,
            )
            self._audio_analyzer.on_partial_result = self._on_partial_record
            print(f"[INIT] ASR model loaded OK", flush=True)
            # 创建屏幕监控实例
            print(f"[INIT] Creating screen monitor (hwnd={self.hwnd}, interval=0.5s) ...", flush=True)
            self._screen_monitor = WindowScreenMonitor(
                hwnd=self.hwnd,
                on_digit_change=self._on_digit_change,
                on_chat_ocr_change=self._on_chat_ocr_change,
                interval=0.5,
            )

            self._audio_analyzer.start()
            self._screen_monitor.start()
            self.is_running = True

        print(f"[INIT] All systems started! Monitoring window: {self.window_title!r}", flush=True)
        print(f"[INIT]   - Screen capture: every 0.5s -> OCR -> detect speaker", flush=True)
        print(f"[INIT]   - Audio capture: default speaker loopback -> Tingwu realtime ASR", flush=True)
        print(f"[INIT]   - Results saved to: data/game_analysis.json", flush=True)
        event_queue.put({"type": "status_change", "data": {"status": "running", "timestamp": iso_now()}})
        return True

    def stop(self):
        if self._backend_mode == "optimized":
            return self.stop_optimized()

        with self._lock:
            if not self.is_running:
                return True
            try:
                if self._screen_monitor:
                    self._screen_monitor.stop()
                if self._audio_analyzer:
                    self._audio_analyzer.stop(round_num=1)
            finally:
                self._screen_monitor = None
                self._audio_analyzer = None
                self.is_running = False
                self._meeting_id = None
                self._partial_text_by_speaker = {}
                self._latest_chat_ocr_text = ""
        event_queue.put({"type": "status_change", "data": {"status": "stopped", "timestamp": iso_now()}})
        return True

    def stop_optimized(self):
        with self._lock:
            if not self.is_running:
                return True
            frame_capture_service = self._optimized_frame_capture_service
            audio_service = self._optimized_audio_service
            audio_thread = self._optimized_audio_thread
            vision_service = self._optimized_vision_service
            coordinator = self._optimized_coordinator

        if frame_capture_service:
            frame_capture_service.stop()
        if audio_service:
            audio_service.stop()
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=3)
        if vision_service:
            vision_service.stop()
        if coordinator and app_state.main_loop:
            fut = asyncio.run_coroutine_threadsafe(coordinator.stop(), app_state.main_loop)
            try:
                fut.result(timeout=5)
            except Exception as e:
                logger.warning("Failed to stop optimized coordinator cleanly: %s", e)

        with self._lock:
            self._optimized_frame_capture_service = None
            self._optimized_audio_service = None
            self._optimized_audio_thread = None
            self._optimized_vision_service = None
            self._optimized_coordinator = None
            self._backend_mode = "legacy"
            self._meeting_id = None
            self._agent = None
            self._current_speaker = None
            self._partial_text_by_speaker = {}
            self._latest_chat_ocr_text = ""
            self.is_running = False

        event_queue.put({"type": "status_change", "data": {"status": "stopped", "timestamp": iso_now()}})
        return True

    def status(self) -> StatusResponse:
        return StatusResponse(
            status="running" if self.is_running else "ready",
            is_running=self.is_running,
            current_speaker=self._current_speaker,
            window_title=self.window_title,
            session_id=self._agent.session_id if self._agent else None,
            meeting_id=self._meeting_id if self.is_running else None,
        )


controller = MonitorController()


@router.get("/")
async def root():
    return FileResponse(_FRONTEND_INDEX, media_type="text/html")


@router.get("/api/status")
async def get_status():
    return controller.status()


@router.post("/api/init")
async def init_system():
    controller.init()
    return {"status": "success", "message": "A侧服务就绪"}


@router.post("/api/debug-ocr")
async def toggle_debug_ocr(enabled: bool = True):
    """Save cropped ROI images to _debug_crops/ for verifying the OCR region."""
    from backend.legacy.extract_speaker_num import enable_debug_save

    enable_debug_save(enabled)
    return {"status": "success", "debug_save": enabled}


@router.post("/api/select-window")
async def api_select_window():
    hwnd, title = await asyncio.get_event_loop().run_in_executor(None, controller.select_window)
    if hwnd is None:
        return {"status": "cancelled"}
    return {"status": "success", "window": {"hwnd": hwnd, "title": title}}


@router.post("/api/new-game")
async def api_new_game(body: NewGameRequest = NewGameRequest()):
    """生成本局 session_id；可选先删除上一局 thread 的 state。"""
    g = app_state.graph
    if g is None:
        raise HTTPException(status_code=503, detail="MemoryGraph 未初始化")
    prev = (body.previous_session_id or "").strip()
    if prev:
        try:
            await g.checkpointer.adelete_thread(prev)
        except Exception as e:
            logger.warning("adelete_thread(%s) failed: %s", prev, e, exc_info=True)
    session_id = f"game_{uuid.uuid4().hex}"
    await ensure_session_state(session_id)
    return {"status": "success", "session_id": session_id}


@router.post("/api/start")
async def api_start(req: StartMonitoringRequest, request: Request):
    sid = req.session_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    meeting_id = f"meeting_{uuid.uuid4().hex[:16]}"
    use_optimized = controller.input_processing_enabled()
    if not use_optimized and controller.hwnd is None:
        await asyncio.get_event_loop().run_in_executor(None, controller.select_window)
        if controller.hwnd is None:
            return {"status": "cancelled", "message": "未选择窗口"}
    try:
        await ensure_session_state(sid)
        if use_optimized:
            await controller.start_optimized(sid, meeting_id, request.app.state)
        else:
            await asyncio.get_event_loop().run_in_executor(None, lambda: controller.start(sid, meeting_id))
        return {
            "status": "success",
            "window_title": controller.window_title,
            "session_id": sid,
            "meeting_id": meeting_id,
            "input_mode": "optimized" if use_optimized else "legacy",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/stop")
async def api_stop():
    await asyncio.get_event_loop().run_in_executor(None, controller.stop)
    return {"status": "success"}


class CropRegionRequest(BaseModel):
    x: int
    y: int
    w: int
    h: int


@router.post("/api/set-crop")
async def api_set_crop(req: CropRegionRequest):
    """Set the sub-region of the window to monitor."""
    if controller._screen_monitor:
        controller._screen_monitor.set_crop_region(req.x, req.y, req.w, req.h)
        return {"status": "success", "crop": {"x": req.x, "y": req.y, "w": req.w, "h": req.h}}
    return {"status": "error", "message": "监控未启动"}


@router.post("/api/clear-crop")
async def api_clear_crop():
    if controller._screen_monitor:
        controller._screen_monitor.clear_crop_region()
    return {"status": "success"}


@router.get("/api/screenshot")
async def api_screenshot():
    """Capture current window and return as JPEG image."""
    if controller.hwnd is None:
        raise HTTPException(status_code=400, detail="未选择窗口")

    def _capture():
        from backend.legacy.screen_monitor import ScreenCapture
        import cv2

        cap = ScreenCapture(controller.hwnd)
        try:
            img = cap.capture()
        finally:
            cap.release()
        if img is None:
            return None
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    jpg_bytes = await asyncio.get_event_loop().run_in_executor(None, _capture)
    if jpg_bytes is None:
        raise HTTPException(status_code=500, detail="截图失败")
    return Response(content=jpg_bytes, media_type="image/jpeg")


@router.post("/api/scan-roster")
async def api_scan_roster(
    session_id: Optional[str] = Query(
        None,
        description="若提供且已有 state，则将名单合并写入该会话的 situation_sketch.player_roster",
    ),
):
    """Take screenshot and extract player roster.

    Prefer local OCR first. If local OCR returns empty, fall back to Gemini.
    """
    if controller.hwnd is None:
        raise HTTPException(status_code=400, detail="未选择窗口")

    def _scan():
        from backend.legacy.screen_monitor import ScreenCapture
        from backend.legacy.extract_speaker_num import extract_player_roster
        from backend.services.gemini_roster import extract_player_roster_gemini

        cap = ScreenCapture(controller.hwnd)
        try:
            img = cap.capture()
        finally:
            cap.release()
        if img is None:
            return {"roster": [], "engine": "capture", "fallback_used": False}

        roster = extract_player_roster(img)
        if roster:
            logger.info("Roster recognized by local OCR: %d players", len(roster))
            return {"roster": roster, "engine": "local_ocr", "fallback_used": False}

        logger.info("Local OCR returned empty roster, falling back to Gemini")
        roster = extract_player_roster_gemini(img)
        return {
            "roster": roster,
            "engine": "gemini" if roster else "local_ocr",
            "fallback_used": True,
        }

    scan_result = await asyncio.get_event_loop().run_in_executor(None, _scan)
    roster = scan_result.get("roster") or []
    if roster:
        controller.roster = roster
        event_queue.put({"type": "roster", "data": roster})
        if session_id:
            try:
                await _merge_gemini_roster_into_state(session_id, roster)
            except Exception as e:
                logger.warning(
                    "merge roster into situation_sketch failed: %s", e, exc_info=True
                )
    return {
        "status": "success",
        "roster": roster,
        "engine": scan_result.get("engine") or "local_ocr",
        "fallback_used": bool(scan_result.get("fallback_used")),
    }


@router.get("/api/roster")
async def api_get_roster():
    return {"roster": controller.roster}


@router.get("/api/model-backend")
async def api_get_model_backend():
    from backend.model.factory import get_current_backend
    return {"backend": get_current_backend()}


@router.post("/api/set-model")
async def api_set_model(backend: str = "api"):
    """Switch LLM backend between 'api' (DashScope) and 'local' (Ollama)."""
    from backend.model.factory import set_model_backend, get_current_backend
    try:
        set_model_backend(backend)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    current = get_current_backend()
    label = "本地 Ollama (qwen2.5:7b)" if current == "local" else "API (通义千问)"
    return {"status": "success", "backend": current, "label": label}


@router.post("/api/tingwu/sync-hotwords")
async def api_tingwu_sync_hotwords():
    manager = TingwuPhraseManager()
    phrase_id = manager.sync_phrase_book(controller._dynamic_name_hotwords())
    status = manager.get_status()
    status.update({"status": "success", "phrase_id": phrase_id})
    return status


@router.get("/api/tingwu/hotwords")
async def api_tingwu_get_hotwords():
    manager = TingwuPhraseManager()
    status = manager.get_status()
    status["status"] = "success"
    return status


class TingwuHotwordsSaveRequest(BaseModel):
    hotwords_text: str = ""


@router.put("/api/tingwu/hotwords")
async def api_tingwu_save_hotwords(req: TingwuHotwordsSaveRequest):
    manager = TingwuPhraseManager()
    status = manager.save_hotwords_text(req.hotwords_text)
    status["status"] = "success"
    return status


class InferenceRequest(BaseModel):
    session_id: str
    self_player_number: int = 1
    self_player_id: str = "player_1"
    role_name: str = "通灵"
    alignment: str = "goose"
    speaker_filter: Optional[list[str]] = None


@router.post("/api/inference/stream")
async def api_inference_stream(req: InferenceRequest):
    """决策流式：SSE，草稿 + 判官 + 修订阶段 token/工具事件，最后 type=done。"""
    from backend.routers.decision import execute_decision_stream

    extra: dict[str, Any] = {}
    if req.speaker_filter:
        extra["speaker_filter"] = req.speaker_filter

    async def _gen():
        try:
            async for ev in execute_decision_stream(
                req.session_id,
                extra=extra or None,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev.get("type") == "done":
                    result = ev.get("result") or {}
                    output = DecisionOutput(
                        session_id=req.session_id,
                        suggestion_type="speak",
                        content=result.get("speech_suggestion", ""),
                        structured={
                            "prior_speech_analysis": result.get(
                                "prior_speech_analysis", ""
                            ),
                            "identity_inference": result.get(
                                "identity_inference", ""
                            ),
                            "warnings": result.get("warnings", []),
                            "debug_prompt": result.get("debug_prompt") or "",
                        },
                        timestamp=iso_now(),
                        trigger="speech",
                    )
                    event_queue.put(
                        {"type": "decision", "data": output.model_dump()}
                    )
        except HTTPException as he:
            err = {"type": "error", "detail": he.detail, "status_code": he.status_code}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"type": "error", "detail": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "status", "data": controller.status().model_dump()})
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)

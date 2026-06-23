from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from backend.app.core.ws_manager import ws_manager
from backend.utils.logger import log

SpeechCallback = Callable[[dict], Awaitable[Any] | Any]


class GGDCoordinator:
    """Fuse audio ASR output with recent vision snapshots to identify speakers."""

    def __init__(
        self,
        vision_service,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        history_limit: int = 200,
        on_speech_result: Optional[SpeechCallback] = None,
    ):
        self.vision = vision_service
        self.vision_history = deque(maxlen=history_limit)
        self.last_speaker_id = -1
        self.current_session: Optional[dict] = None
        self.loop = loop or asyncio.get_event_loop()
        self.task_queue: asyncio.Queue[dict] = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.on_speech_result = on_speech_result
        self.worker_task = self.loop.create_task(self.process_task())
        log.success("GGD coordinator started.")

    def record_vision_state(self, res: dict) -> None:
        self.vision_history.append(
            {
                "ts": time.time(),
                "seat": res.get("active_seat"),
                "ui": res.get("ui_text", ""),
            }
        )

    def on_audio_result(
        self,
        text: str,
        start_time: float,
        duration: float,
        asr_spk: str,
    ) -> None:
        if not text:
            return
        self.loop.call_soon_threadsafe(
            self.task_queue.put_nowait,
            {
                "text": text,
                "start_time": start_time,
                "duration": duration,
                "asr_spk": asr_spk,
            },
        )

    async def process_task(self) -> None:
        log.info("GGD coordinator speech fusion worker started.")
        while True:
            item = await self.task_queue.get()
            try:
                speaker_info = await self.loop.run_in_executor(
                    self.executor,
                    self._determine_speaker,
                    item["start_time"],
                    item["duration"],
                    item["asr_spk"],
                )
                seat_id = speaker_info["seat_id"]
                if seat_id == self.last_speaker_id and self.current_session:
                    self.current_session["content"] += f"    {item['text']}"
                    self.current_session["type"] = "update"
                else:
                    self.current_session = {
                        "type": "new",
                        "id": int(time.time() * 1000),
                        "timestamp": round(item["start_time"], 2),
                        "seat_id": seat_id,
                        "name": speaker_info["name"],
                        "content": item["text"],
                        "method": speaker_info["method"],
                    }
                    self.last_speaker_id = seat_id
                await self._dispatch_result(self.current_session)
            except Exception as e:
                log.error(f"GGD coordinator failed to process audio result: {e}")
            finally:
                self.task_queue.task_done()

    def _determine_speaker(
        self,
        start_ts: float,
        duration: float,
        asr_spk: str,
    ) -> dict:
        end_ts = start_ts + duration
        snapshots = [
            s
            for s in self.vision_history
            if start_ts - 0.2 <= s["ts"] <= end_ts + 0.2
        ]

        for snapshot in reversed(snapshots):
            match = re.search(r"(\d+)(?:发言|鍙戣█)", snapshot["ui"])
            if match:
                seat_id = int(match.group(1))
                return {
                    "seat_id": seat_id,
                    "name": self.vision.seat_names.get(seat_id, "Unknown"),
                    "method": "ui_ocr",
                }

        seat_votes: dict[int, int] = {}
        for snapshot in snapshots:
            if snapshot["seat"]:
                seat_votes[snapshot["seat"]] = seat_votes.get(snapshot["seat"], 0) + 1

        if seat_votes:
            seat_id = max(seat_votes, key=seat_votes.get)
            return {
                "seat_id": seat_id,
                "name": self.vision.seat_names.get(seat_id, "Unknown"),
                "method": "hsv_vision",
            }

        return {
            "seat_id": None,
            "name": f"Unknown({asr_spk})",
            "method": "audio_engine",
        }

    async def _dispatch_result(self, payload: dict) -> None:
        log.success(
            f"{payload.get('seat_id')} - {payload.get('name')} speech: "
            f"{payload.get('content', '')[:30]}..."
        )
        try:
            await ws_manager.broadcast(payload)
        except Exception as e:
            log.error(f"Failed to broadcast optimized input message: {e}")

        if self.on_speech_result:
            try:
                result = self.on_speech_result(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                log.error(f"Optimized input speech callback failed: {e}")

    async def stop(self) -> None:
        log.info("Stopping GGD coordinator.")
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)

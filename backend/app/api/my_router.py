from __future__ import annotations

import base64
import json
from typing import Any

import cv2
from fastapi import APIRouter, Body, HTTPException, Request, WebSocket, WebSocketDisconnect

from backend.app.core.ws_manager import ws_manager
from backend.utils.config_loader import get_public_config, save_public_config
from backend.utils.logger import log
from backend.utils.path_tool import get_abs_path

router = APIRouter()


@router.get("/perception-status")
async def get_service_status():
    return {"status": "online", "agent": "GGD_Perception_Agent"}


@router.get("/config")
async def get_config():
    return {"status": "success", "config": get_public_config()}


def _apply_runtime_config(request: Request, saved_config: dict[str, Any]) -> None:
    vision_config = saved_config.get("vision", {})
    frame_capture_service = getattr(request.app.state, "frame_capture_service", None)
    if frame_capture_service:
        frame_capture_service.update_config(
            vision_config.get("mode"),
            vision_config.get("target"),
            vision_config.get("fps_limit"),
        )

    game_setting = saved_config.get("game_setting", {})
    vision_service = getattr(request.app.state, "vision_service", None)
    if vision_service and "seat_num" in game_setting:
        vision_service.set_seat_num(game_setting["seat_num"])


@router.put("/config")
async def update_config(request: Request, config: dict = Body(...)):
    try:
        current_config = get_public_config()
        saved_config = save_public_config(config)
        _apply_runtime_config(request, saved_config)
        restart_required = (
            current_config.get("server", {}).get("host")
            != saved_config.get("server", {}).get("host")
            or current_config.get("server", {}).get("port")
            != saved_config.get("server", {}).get("port")
        )
        return {
            "status": "success",
            "config": saved_config,
            "restart_required": restart_required,
            "message": (
                "Configuration saved. Runtime capture settings were updated."
                if not restart_required
                else "Configuration saved. Host/port changes require restarting the app."
            ),
        }
    except Exception as e:
        log.error(f"Failed to save input-processing config: {e}")
        return {"status": "error", "message": str(e)}


@router.websocket("/ws/stream")
async def vision_websocket(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") != "request_capture_frame":
                continue

            capture_service = getattr(websocket.app.state, "frame_capture_service", None)
            if capture_service is None:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "input-processing capture service is not running",
                    }
                )
                continue

            frame = capture_service.get_latest_frame()
            if frame is None:
                await websocket.send_json(
                    {"type": "error", "message": "no frame has been captured yet"}
                )
                continue

            _, buffer = cv2.imencode(".jpg", frame)
            base64_str = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_json(
                {
                    "type": "calibration_frame",
                    "image": f"data:image/jpeg;base64,{base64_str}",
                }
            )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        log.error(f"Input-processing WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@router.post("/calibrate")
async def save_calibration(request: Request, config: dict = Body(...)):
    vision_service = getattr(request.app.state, "vision_service", None)
    if vision_service is None:
        raise HTTPException(
            status_code=409,
            detail="input-processing vision service is not running",
        )

    try:
        config_path = get_abs_path("roi_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        updated_data = vision_service.update_config()
        return {
            "status": "success",
            "config_preview": updated_data.get("config_preview") if updated_data else None,
        }
    except Exception as e:
        log.error(f"Failed to save calibration: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/calibration/current")
async def get_current_calibration(request: Request):
    vision_service = getattr(request.app.state, "vision_service", None)
    if vision_service is None:
        return {
            "status": "not_running",
            "config_preview": None,
            "message": "input-processing vision service is not running",
        }

    try:
        current_data = vision_service.update_config()
        return {
            "status": "success",
            "config_preview": current_data.get("config_preview") if current_data else None,
        }
    except Exception as e:
        log.error(f"Failed to load calibration: {e}")
        return {"status": "error", "message": str(e)}

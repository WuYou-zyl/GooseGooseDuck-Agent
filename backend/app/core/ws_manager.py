from fastapi import WebSocket
from typing import List
from backend.utils.logger import log

class ConnectionManager:
    def __init__(self):
        # 存放活跃的 WebSocket 连接
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"新客户端连接，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            log.info("客户端断开连接")

    async def broadcast(self, message: dict):
        """向所有连接的客户端发送 JSON 消息"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                log.error(f"发送消息失败: {e}")
                # 如果发送失败，尝试清理无效连接
                self.active_connections.remove(connection)

# 导出单例
ws_manager = ConnectionManager()
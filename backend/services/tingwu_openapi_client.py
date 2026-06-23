from __future__ import annotations

import json
import os
from typing import Any

from aliyunsdkcore.http import protocol_type


class TingwuOpenApiClient:
    def __init__(self) -> None:
        self._acs_client = None

    def create_phrase(self, body: dict[str, Any]) -> str:
        request = self._create_common_request("POST", self._phrase_path())
        request.set_content(json.dumps(body).encode("utf-8"))
        data = self._do_action(request)
        return self._extract_phrase_id(data)

    def update_phrase(self, phrase_id: str, body: dict[str, Any]) -> dict[str, Any]:
        request = self._create_common_request(
            "PUT", f"{self._phrase_path()}/{phrase_id}"
        )
        request.set_content(json.dumps(body).encode("utf-8"))
        return self._do_action(request)

    def list_phrases(self) -> list[dict[str, Any]]:
        request = self._create_common_request("GET", self._phrase_path())
        data = self._do_action(request)
        payload = data.get("Data")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("Phrases", "PhraseList", "Items", "List"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        return []

    def create_realtime_task(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self._create_common_request("PUT", self._task_path())
        request.add_query_param("type", "realtime")
        request.set_content(json.dumps(body).encode("utf-8"))
        return self._do_action(request)

    def stop_realtime_task(self, task_id: str) -> dict[str, Any]:
        body = {
            "AppKey": self._tingwu_app_key(),
            "Input": {"TaskId": task_id},
        }
        request = self._create_common_request("PUT", self._task_path())
        request.add_query_param("type", "realtime")
        request.add_query_param("operation", "stop")
        request.set_content(json.dumps(body).encode("utf-8"))
        return self._do_action(request)

    def get_task_info(self, task_id: str) -> dict[str, Any]:
        request = self._create_common_request("GET", f"{self._task_path()}/{task_id}")
        return self._do_action(request)

    def _create_common_request(self, method: str, uri: str):
        from aliyunsdkcore.request import CommonRequest

        request = CommonRequest()
        request.set_accept_format("json")
        request.set_domain(self._tingwu_domain())
        request.set_version(self._tingwu_version())
        request.set_protocol_type(self._resolve_protocol_type(self._tingwu_protocol()))
        request.set_method(method)
        request.set_uri_pattern(uri)
        request.add_header("Content-Type", "application/json")
        return request

    def _do_action(self, request) -> dict[str, Any]:
        response = self._get_acs_client().do_action_with_exception(request)
        if isinstance(response, bytes):
            return json.loads(response.decode("utf-8"))
        if isinstance(response, str):
            return json.loads(response)
        if isinstance(response, dict):
            return response
        raise TypeError(f"Unsupported Tingwu response type: {type(response)!r}")

    def _get_acs_client(self):
        if self._acs_client is not None:
            return self._acs_client

        access_key_id = _env_first("ALIBABA_CLOUD_ACCESS_KEY_ID", "ALIYUN_ACCESS_KEY_ID")
        access_key_secret = _env_first(
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
            "ALIYUN_ACCESS_KEY_SECRET",
        )
        if not access_key_id or not access_key_secret:
            raise ValueError("Aliyun AccessKey credentials are not configured")

        from aliyunsdkcore.auth.credentials import AccessKeyCredential
        from aliyunsdkcore.client import AcsClient

        credentials = AccessKeyCredential(access_key_id, access_key_secret)
        self._acs_client = AcsClient(
            region_id=_env_first("TINGWU_REGION_ID") or "cn-beijing",
            credential=credentials,
        )
        return self._acs_client

    @staticmethod
    def _resolve_protocol_type(raw_protocol: str) -> str:
        normalized = raw_protocol.strip().lower()
        if normalized == protocol_type.HTTPS:
            return protocol_type.HTTPS
        if normalized == protocol_type.HTTP:
            return protocol_type.HTTP
        raise ValueError(f"Unsupported Tingwu protocol: {raw_protocol!r}")

    def _tingwu_app_key(self) -> str:
        value = _env_first("TINGWU_APP_KEY")
        if not value:
            raise ValueError("TINGWU_APP_KEY is not configured")
        return value

    def _tingwu_domain(self) -> str:
        return _env_first("TINGWU_DOMAIN") or "tingwu.cn-beijing.aliyuncs.com"

    def _tingwu_version(self) -> str:
        return _env_first("TINGWU_VERSION") or "2023-09-30"

    def _tingwu_protocol(self) -> str:
        return _env_first("TINGWU_PROTOCOL") or "https"

    def _task_path(self) -> str:
        return _env_first("TINGWU_TASK_PATH") or "/openapi/tingwu/v2/tasks"

    def _phrase_path(self) -> str:
        return _env_first("TINGWU_PHRASE_PATH") or "/openapi/tingwu/v2/resources/phrases"

    @staticmethod
    def _extract_phrase_id(data: dict[str, Any]) -> str:
        payload = data.get("Data")
        if isinstance(payload, dict):
            for key in ("PhraseId", "Id", "ResourceId"):
                value = payload.get(key)
                if value:
                    return str(value)
        for key in ("PhraseId", "Id", "ResourceId"):
            value = data.get(key)
            if value:
                return str(value)
        return ""


def _env_first(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""

"""``ChatGemini`` — a LangChain chat model backed by the Gemini REST API.

Implemented as a first-class ``BaseChatModel`` (no google SDK) so it drops into
LCEL chains, agents, and ``.invoke()`` / ``.stream()`` like any other LangChain
model. Works with the AQ.* keys that the SDK rejects, and degrades to a clearly
labelled offline stub when no key is configured (keeps CI and local dev honest).
"""
from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_GENAI = "https://generativelanguage.googleapis.com/v1beta"


class ChatGemini(BaseChatModel):
    api_key: str = ""
    model: str = "gemini-2.5-flash"
    timeout: float = 45.0
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _to_gemini(self, messages: list[BaseMessage]) -> tuple[str | None, list]:
        system: str | None = None
        contents: list = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system = str(m.content)
            elif isinstance(m, AIMessage):
                contents.append({"role": "model", "parts": [{"text": str(m.content)}]})
            else:
                contents.append({"role": "user", "parts": [{"text": str(m.content)}]})
        return system, contents

    def _payload(self, messages: list[BaseMessage]) -> tuple[str, dict]:
        system, contents = self._to_gemini(messages)
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return system or "", payload

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        if not self.configured:
            text = "[offline mode] Set GEMINI_API_KEY to enable grounded generation."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
        _, payload = self._payload(messages)
        url = f"{_GENAI}/models/{self.model}:generateContent?key={self.api_key}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        if not self.configured:
            yield ChatGenerationChunk(message=AIMessageChunk(content="[offline mode]"))
            return
        _, payload = self._payload(messages)
        url = f"{_GENAI}/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if raw in ("", "[DONE]"):
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for cand in data.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            piece = part.get("text", "")
                            if piece:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=piece)
                                )
                                if run_manager:
                                    run_manager.on_llm_new_token(piece, chunk=chunk)
                                yield chunk

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Convenience helper used by the agent loop."""
        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        return self._generate(messages).generations[0].message.content

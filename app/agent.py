"""A small, testable ReAct-style tool-calling agent.

The agent prompts an LLM to emit a JSON action each turn — either call a tool or
return a final answer — and an executor runs the loop. The LLM is injected as a
plain ``generate(prompt) -> str`` callable, which keeps the executor fully
testable without a network: tests pass a scripted callable and assert the tool
actually ran. In production the callable is ``ChatGemini.complete``.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field

# --- tools -----------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_calculator(expr: str) -> str:
    """Evaluate an arithmetic expression without ``eval`` (AST whitelist only)."""

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        value = _eval(ast.parse(expr, mode="eval").body)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as an observation
        return f"error: {exc}"
    return str(value)


Tool = Callable[[str], str]


@dataclass
class AgentStep:
    thought: str
    tool: str
    tool_input: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)


_SYSTEM = """You are a tool-using assistant. Think step by step.
On each turn reply with ONE JSON object and nothing else.
To use a tool: {"thought": "...", "tool": "<name>", "tool_input": "..."}
To finish:     {"thought": "...", "answer": "..."}
Available tools:
{tools}
"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse_action(raw: str) -> dict:
    match = _JSON.search(raw)
    if not match:
        return {"answer": raw.strip()}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"answer": raw.strip()}


class Agent:
    def __init__(self, generate: Callable[[str], str], tools: dict[str, Tool], max_steps: int = 5):
        self.generate = generate
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        tool_docs = "\n".join(f"- {name}" for name in self.tools)
        # Plain replace, not str.format: the template contains literal JSON braces.
        system = _SYSTEM.replace("{tools}", tool_docs)
        transcript = f"Question: {question}"
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            raw = self.generate(f"{system}\n\n{transcript}\n\nJSON:")
            action = _parse_action(raw)
            if "answer" in action and "tool" not in action:
                return AgentResult(answer=str(action["answer"]).strip(), steps=steps)
            tool = str(action.get("tool", ""))
            tool_input = str(action.get("tool_input", ""))
            if tool not in self.tools:
                observation = f"error: unknown tool '{tool}'"
            else:
                observation = self.tools[tool](tool_input)
            steps.append(
                AgentStep(
                    thought=str(action.get("thought", "")),
                    tool=tool,
                    tool_input=tool_input,
                    observation=observation,
                )
            )
            transcript += (
                f'\n{json.dumps({"tool": tool, "tool_input": tool_input})}'
                f"\nObservation: {observation}"
            )

        return AgentResult(answer="Stopped: step budget exhausted.", steps=steps)

"""The agent executor runs tools driven by a scripted (network-free) LLM."""
from app.agent import Agent, safe_calculator


def test_safe_calculator_evaluates():
    assert safe_calculator("2 + 3 * 4") == "14"
    assert safe_calculator("(10 - 4) / 2") == "3.0"


def test_safe_calculator_rejects_code_injection():
    # Must never execute arbitrary Python.
    assert safe_calculator("__import__('os').system('echo hi')").startswith("error")
    assert safe_calculator("open('x')").startswith("error")


def test_agent_calls_tool_then_answers():
    # A scripted LLM: first turn calls the calculator, second turn answers.
    script = [
        '{"thought": "compute", "tool": "calculator", "tool_input": "6 * 7"}',
        '{"thought": "done", "answer": "The result is 42."}',
    ]
    calls = iter(script)

    def fake_llm(_prompt: str) -> str:
        return next(calls)

    agent = Agent(generate=fake_llm, tools={"calculator": safe_calculator})
    result = agent.run("What is 6 times 7?")

    assert result.answer == "The result is 42."
    assert len(result.steps) == 1
    assert result.steps[0].tool == "calculator"
    assert result.steps[0].observation == "42"


def test_agent_handles_unknown_tool_gracefully():
    script = [
        '{"tool": "teleport", "tool_input": "moon"}',
        '{"answer": "Cannot teleport."}',
    ]
    calls = iter(script)
    agent = Agent(generate=lambda _p: next(calls), tools={"calculator": safe_calculator})
    result = agent.run("Take me to the moon")
    assert "unknown tool" in result.steps[0].observation
    assert result.answer == "Cannot teleport."


def test_agent_respects_step_budget():
    # An LLM that always calls a tool and never answers must terminate.
    agent = Agent(
        generate=lambda _p: '{"tool": "calculator", "tool_input": "1+1"}',
        tools={"calculator": safe_calculator},
        max_steps=3,
    )
    result = agent.run("loop forever")
    assert len(result.steps) == 3
    assert "budget" in result.answer.lower()

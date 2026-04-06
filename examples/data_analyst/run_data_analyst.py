"""
Data Analyst Agent — ToolSage python_repl demo.

Runs a LangGraph ReAct agent that answers sales analysis questions by writing
and executing Python code. After the agent finishes, scores all tool calls
then proposes manifest improvements based on divergence patterns.

Run:
    ANTHROPIC_API_KEY=... python examples/data_analyst/run_data_analyst.py
"""

from agent import build_agent
from data import ANALYSIS_TASK
from tools import sage


def main():
    agent = build_agent()

    print("=" * 70)
    print("RUNNING AGENT — 2024 Sales Data Analysis")
    print("=" * 70)
    print()

    sage.set_task(ANALYSIS_TASK)
    result = agent.invoke({"messages": [{"role": "user", "content": ANALYSIS_TASK}]})

    print("\nAGENT RESPONSE:")
    print("=" * 70)
    print(result["messages"][-1].content)
    print("=" * 70)

    print()
    sage.score()
    sage.improve()


if __name__ == "__main__":
    main()

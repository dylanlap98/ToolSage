"""
AI/ML History Agent — ToolSage augmentation layer demo.

Uses a LangGraph ReAct agent with a Wikipedia search tool wrapped by @sage.tool.
The manifest injects guidance into the tool's description at decoration time,
before the agent ever makes a call.

Run:
    ANTHROPIC_API_KEY=... python examples/ai_history_agent.py
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wikipedia
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from toolsage import ToolSage
from utilities.env import populate_env

populate_env()  # loads secrets-manifest.env → secrets.json → os.environ

# ---------------------------------------------------------------------------
# 1. Set up ToolSage and define the tool
# ---------------------------------------------------------------------------

sage = ToolSage()

MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "manifests", "wikipedia_search.manifest.md"
)

@tool
@sage.tool(MANIFEST_PATH)
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a summary of the most relevant article."""
    try:
        return wikipedia.summary(query, sentences=4, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"DisambiguationError: '{query}' is ambiguous. Options include: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"PageError: No Wikipedia article found for '{query}'."


# ---------------------------------------------------------------------------
# 2. Show the augmentation — print the tool description before running
# ---------------------------------------------------------------------------

def print_augmented_description():
    print("=" * 70)
    print("TOOLSAGE AUGMENTATION LAYER")
    print("Tool description seen by the agent:")
    print("=" * 70)
    print(wikipedia_search.description)
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# 3. Build and run the LangGraph agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a historian of artificial intelligence and machine learning.
Your job is to research and narrate landmark moments in AI/ML history using Wikipedia.
For each topic you research, extract the key year, the people involved, and why it mattered.
Build a coherent narrative across the topics you investigate."""

RESEARCH_TASK = """Research the following landmarks in AI/ML history and tell the story of how they connect:

1. The Turing Test — where did the idea of machine intelligence begin?
2. The Perceptron — the first trainable neural network
3. Geoffrey Hinton — the researcher who brought neural networks back from the dead
4. The ImageNet competition — the moment deep learning proved itself
5. The Transformer architecture — the paper that changed everything

Use the wikipedia_search tool for each topic. Then synthesize a short narrative connecting these moments."""


def run_agent():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    agent = create_react_agent(llm, tools=[wikipedia_search], prompt=SYSTEM_PROMPT)

    print("RUNNING AGENT — AI/ML History Research\n")

    result = agent.invoke({"messages": [{"role": "user", "content": RESEARCH_TASK}]})

    final_message = result["messages"][-1].content
    print("\nAGENT RESPONSE:")
    print("=" * 70)
    print(final_message)
    print("=" * 70)


if __name__ == "__main__":
    print_augmented_description()
    run_agent()

"""
LangGraph ReAct agent for data analysis.
Receives a task, uses python_repl to compute answers, synthesizes a response.
"""

import warnings

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import ToolNode, create_react_agent

from tools import python_repl

SYSTEM_PROMPT = """\
You are a data analyst. Answer questions by writing and running Python code
using the python_repl tool.

The variable SALES_DATA is available in scope when your code runs. It contains:
  - SALES_DATA["months"]     — list of 12 month name strings
  - SALES_DATA["revenue"]    — list of 12 monthly revenue integers (USD)
  - SALES_DATA["units_sold"] — list of 12 monthly unit count integers
  - SALES_DATA["region"]     — string
  - SALES_DATA["year"]       — int

The statistics module is also available. No other third-party packages are available.

Use one tool call per question. Always print your results.\
"""


def build_agent():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    tool_node = ToolNode([python_repl], handle_tool_errors=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_react_agent(llm, tools=tool_node, prompt=SYSTEM_PROMPT)

"""
agent/graph.py — LangGraph agent for the Senior Economist
=========================================================
Defines a conversational agent powered by DeepSeek (via OpenAI-compatible API)
that uses LangGraph to reason about the GDP nowcast data and decide which tools
to call.

The agent uses a ReAct-style loop:
  1. LLM decides which tool to call (or responds directly)
  2. Tool is executed
  3. Result is sent back to the LLM
  4. Loop until the LLM provides a final answer
"""

import os
import json
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from openai import OpenAI

# ── Import our tool functions ────────────────────────────────────────────────
from agent.tools import (
    get_nowcast,
    get_model_comparison,
    get_feature_importance,
    get_gdp_trend,
    get_data_summary,
)


# ── Wrap tools for LangChain ─────────────────────────────────────────────────

@tool
def nowcast_tool() -> str:
    """Get the current GDP nowcast for all 5 models (OLS, Ridge, SVR, Gradient Boosting, Random Forest) for the latest month."""
    return get_nowcast()


@tool
def model_comparison_tool() -> str:
    """Get a detailed comparison table of all 5 ML models with nowcast, error, and CV MAE metrics."""
    return get_model_comparison()


@tool
def feature_importance_tool() -> str:
    """Get the top features driving the Random Forest GDP prediction model."""
    return get_feature_importance()


@tool
def gdp_trend_tool(period: str = "all") -> str:
    """Get historical GDP trend information. Period can be 'all' (full history), '5y' (5 years), '10y' (10 years), or '3y' (3 years)."""
    return get_gdp_trend(period)


@tool
def data_summary_tool() -> str:
    """Get a summary of the training dataset: rows, features, date range, model info."""
    return get_data_summary()


# ── Register tools ───────────────────────────────────────────────────────────
tools = [nowcast_tool, model_comparison_tool, feature_importance_tool, gdp_trend_tool, data_summary_tool]
tool_map = {t.name: t for t in tools}


# ── System prompt for the Senior Economist persona ──────────────────────────
SYSTEM_PROMPT = """You are a Senior Economist at the Bank of Canada, specialising in GDP nowcasting.

You have access to a machine learning nowcast system that predicts Canadian GDP
(chain 2017 dollars, seasonally adjusted at annual rates) using employment,
CPI, and manufacturing sales. The system uses 5 models:
- OLS Regression (baseline)
- Ridge Regression
- SVR with RBF kernel
- Gradient Boosting
- Random Forest (primary model)

YOUR PERSONALITY:
- You are professional, data-driven, and insightful.
- You explain economic concepts clearly without being condescending.
- You always cite specific numbers and model names when answering.
- You provide context and interpretation, not just raw data.
- When asked about trends, you note whether changes are economically significant.

RULES:
- Use the tools available to you to answer questions. Do not make up numbers.
- If the user asks a question you cannot answer with the available tools, tell them honestly.
- Keep responses concise but informative. Use bullet points for clarity.
- Always highlight the best-performing model when discussing model results.

You may greet the user briefly when first addressed, then wait for their question.
"""


# ── DeepSeek client ──────────────────────────────────────────────────────────

def _get_client():
    """Return an OpenAI client configured for DeepSeek."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not set. "
            "Create a .env file with DEEPSEEK_API_KEY=your_key or set it as an environment variable."
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")


def _call_deepseek(messages: list, tools_spec: list = None) -> dict:
    """Call the DeepSeek chat API and return the response dict."""
    client = _get_client()
    kwargs = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    if tools_spec:
        kwargs["tools"] = tools_spec
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.model_dump()


# ── Tools specification for DeepSeek ─────────────────────────────────────────

def _get_tools_spec() -> list:
    """Build the tools specification for the DeepSeek API."""
    tools_spec = [
        {
            "type": "function",
            "function": {
                "name": "nowcast_tool",
                "description": "Get the current GDP nowcast for all 5 models for the latest month.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "model_comparison_tool",
                "description": "Get a detailed comparison table of all 5 ML models with nowcast, error, and CV MAE metrics.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "feature_importance_tool",
                "description": "Get the top features driving the Random Forest GDP prediction model.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "gdp_trend_tool",
                "description": "Get historical GDP trend information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string",
                            "enum": ["all", "5y", "10y", "3y"],
                            "description": "Time period: 'all' (full history), '5y' (5 years), '10y' (10 years), or '3y' (3 years)",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "data_summary_tool",
                "description": "Get a summary of the training dataset: rows, features, date range, model info.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    return tools_spec


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Agent node ───────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    LLM reasoning node.
    Builds API-compatible messages from state and calls DeepSeek.
    """
    # Build API-compatible message list from the conversation state
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in state["messages"]:
        if isinstance(msg, SystemMessage):
            api_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            api_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                # Single assistant message with all tool_calls
                tcs = []
                for tc in msg.tool_calls:
                    tcs.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    })
                api_messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": tcs,
                })
            else:
                api_messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            api_messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": str(msg.content),
            })

    # Call DeepSeek
    tools_spec = _get_tools_spec()
    response = _call_deepseek(api_messages, tools_spec)

    # Parse response
    content = response.get("content", "")
    tool_calls_raw = response.get("tool_calls", [])

    ai_msg = AIMessage(content=content or "")

    if tool_calls_raw:
        ai_msg.tool_calls = []
        for tc in tool_calls_raw:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            ai_msg.tool_calls.append({
                "name": tc["function"]["name"],
                "args": args,
                "id": tc.get("id", f"call_{len(ai_msg.tool_calls)}"),
            })

    return {"messages": [ai_msg]}


# ── Tools node ───────────────────────────────────────────────────────────────

def tools_node(state: AgentState) -> dict:
    """Execute tool calls requested by the LLM."""
    last_msg = state["messages"][-1]
    tool_messages = []

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {"messages": []}

    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "call_1")

        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Error executing {tool_name}: {str(e)}"
        else:
            result = f"Unknown tool: {tool_name}"

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

    return {"messages": tool_messages}


# ── Conditional routing ──────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["tools", "final"]:
    """Determine whether to continue to tools or end."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "final"


# ── Build the graph ──────────────────────────────────────────────────────────

def build_agent():
    """Build and compile the Senior Economist LangGraph agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "final": END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ── Convenience runner ───────────────────────────────────────────────────────

def run_agent(user_query: str, history: list = None) -> tuple[str, list]:
    """
    Run the agent with a user query and optional conversation history.

    Args:
        user_query: The user's question.
        history: List of previous (role, content) tuples, e.g.
                 [("user", "hello"), ("assistant", "hi")]

    Returns:
        (response_text, updated_history)
    """
    if history is None:
        history = []

    graph = build_agent()

    # Build message list from history
    messages = []
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_query))

    # Run the graph
    config = {"recursion_limit": 25}
    result = graph.invoke({"messages": messages}, config)

    # Extract the final AI response and build updated history
    final_response = ""
    updated_history = list(history)  # start with previous history
    updated_history.append(("user", user_query))

    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            final_response = msg.content

    if final_response:
        updated_history.append(("assistant", final_response))

    return final_response, updated_history
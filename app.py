# System modules
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Tuple, Dict, TypedDict, Literal

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.types import Command

load_dotenv()
SAVE_DIR = 'charts'
os.makedirs(SAVE_DIR, exist_ok=True)
model = ChatOpenAI(model="gpt-4o-mini")


@tool(return_direct=True, response_format="content_and_artifact")
def fetch_stock_data(tickers: List[str], period='1y') -> Tuple[str, dict]:
    """
    Fetch historical stock data for a list of tickers over a specified period.

    Args:
        tickers: A list of stock ticker symbols.
        period: The period over which to fetch the stock data (default is '1y').
    
    Returns:
        dict: A dictionary where the keys are ticker symbols and the values are the historical stock data.
    """
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        data[ticker] = history.reset_index().to_dict(orient='list')
    response = f"The data for tickers: {tickers} retrieved successfully for period: {period}"
    return response, data

def compute_metrics(data: dict) -> dict:
    """
    Computes the Annual returns and Volatility metrics from the data.

    Args:
        data: The Data dictionary
    
    Returns:
        results: A dictionary of computed metrics
    """
    results = {}
    for ticker, history in data.items():
        # Convert history to DataFrame
        df = pd.DataFrame.from_dict(history)

        # Check for 'Close' prices
        if 'Close' not in df.columns or df['Close'].dropna().empty:
            print(f"Warning: No closing price data for {ticker}")
            continue
        
        # Calculate metrics
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        annual_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Store results
        results[ticker] = {
            "Annual Return": float(annual_return),   # Convert to Python float
            "Volatility": float(volatility)         # Convert to Python float
        }
    return results

def visualize_trends(data: Dict) -> str:
    """ 
    Visualize the closing price trends for multiple stocks.

    Args:
        data: A dictionary containing the ticker symbols and their corresponding historical stock data.

    Returns:
        None: It displays a matplotlib chart.
    """
    if not data:
        print("No data to visualize.")
        return 
    
    # Create the figure
    plt.figure(figsize=(12, 6))
    for ticker, history in data.items():
        plt.plot(history['Date'], history['Close'], label=ticker)

    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.title('Stock Price Trends')
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{SAVE_DIR}/stock_trend_{timestamp}.png"

    plt.savefig(file_name)
    plt.close()

    return f"Chart saved to: {file_name}"


data_retriever_agent = model.bind_tools([fetch_stock_data])

members = ["data_retriever", "analyzer", "visualizer", "allocator"]
options = members + ["FINISH"]

class AgentState(MessagesState):
    next: str
    data: dict
    analysis: dict 
    summary: str

class Router(TypedDict):
    next: Literal[*options] # type: ignore


SUPERVISOR_PROMPT = """
You are a supervisor tasked with managing a conversation between the following workers: {members}. 
Given the current state and following user request, respond with the worker that needs to act next. 
Select the workers that are needed based on the feedback:
- "data_retriever" (to fetch the relevant stock data)
- "visualizer" (to visualize the fetched data)
- "analyzer" (to analyze the trends from the data)
- "allocator" (to suggest fund allocation in the sector)
Each worker will perform a task and respond with their results and status.
DO NOT CALL any agent MORE THAN ONCE.
When finished, respond with FINISH."""

DATA_RETRIEVER_PROMPT = """
You are a data retriever. You must call the fetch_stock_data tool to retrieve the historical stock data which the user requires.The tool argument should be LIST OF TICKERS given by user. ONLY OUTPUT tool_calls AND NOTHING ELSE.
"""

ANALYZER_PROMPT = """
You are the Stock Data Analyzer.Your task is to interpret stock performance metrics (annual return and volatility) given by the user to identify trends.

For each stock, classify the trend as one of the following:
- **Bullish:** Positive returns with moderate volatility.
- **Bearish:** Negative returns with high volatility.
- **Stable:** Low volatility with flat or minimal returns.
- **Speculative:** Positive returns with very high volatility.
If volatility is extremely high, flag the stock as high risk. Only return a short summary for each stock, without explanations or allocation recommendations.
"""

ALLOCATOR_PROMPT = """
You are the Fund Allocator. Your job is to recommend optimal fund allocations based on the findings and calculations given by the user.
"""


def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]: # type: ignore
    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT.format(members=members)),
    ] + state["messages"]
    response = model.with_structured_output(Router, method="function_calling").invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(update={"next": goto}, goto=goto)


def data_retriever_node(state: AgentState) -> Command[Literal["supervisor"]]:
    data = {}
    messages = [
        SystemMessage(content=DATA_RETRIEVER_PROMPT),
    ] + state["messages"]
    ai_msg = data_retriever_agent.invoke(messages)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"fetch_stock_data": fetch_stock_data}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        data = tool_msg.artifact
        messages.append(tool_msg)
    
    return Command(
        update={
            "messages": [HumanMessage(content=messages[-1].content, name="data_retriever")],
            "data": data,
        },
        goto="supervisor"
    )


def visualizer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    data = state["data"]
    response = visualize_trends(data=data)
    
    return Command(
        update={
            "messages": [HumanMessage(content=response, name="visualizer")],
        },
        goto="supervisor"
    )


def analyzer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    data = state["data"]
    metrics = compute_metrics(data)
    messages = [
        SystemMessage(content=ANALYZER_PROMPT),
        HumanMessage(content=f"Metrics:\n {metrics}"),
    ] + state["messages"]
    response = model.invoke(messages)
    return Command(
        update={
            "messages": [HumanMessage(content=response.content, name="analyzer")],
            "analysis": response.content
        },
        goto="supervisor"
    )


def allocator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    analysis = state["analysis"]
    messages = [
        SystemMessage(content=ALLOCATOR_PROMPT),
        HumanMessage(content=f"Analysis: {analysis}"),
    ] + state["messages"]
    response = model.invoke(messages)
    return Command(
        update={
            "messages": [HumanMessage(content=response.content, name="allocator")],
            "summary": response.content,
        },
        goto="supervisor"
    )


graph_builder = StateGraph(AgentState)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("data_retriever", data_retriever_node)
graph_builder.add_node("visualizer", visualizer_node)
graph_builder.add_node("analyzer", analyzer_node)
graph_builder.add_node("allocator", allocator_node)
graph_builder.set_entry_point("supervisor")

graph = graph_builder.compile()


for s in graph.stream(
    {"messages": [HumanMessage(content="Help me decide how to distribute investments across META, MSFT, and AMD.")]}, subgraphs=True
):
    print(s)
    print("-"*50)
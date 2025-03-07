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
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

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


@tool
def fetch_stocks_by_risk(risk_level: str) -> str:
    """
    Recommend stocks based on the user's risk tolerance.
    
    Args:
        risk_level: The user's risk tolerance (low, medium, high)
        
    Returns:
        String of list of ticker symbols.
    """
    recommendations = {
        "low": ["JNJ", "KO", "PG"],
        "medium": ["AAPL", "MSFT", "GOOGL"],
        "high": ["TSLA", "NVDA", "AMZN"]
    }
    recommended_stocks = recommendations.get(risk_level.lower(), [])
    return f"Here are the stocks for {risk_level.lower()} risk level: {recommended_stocks}."


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
recommender_agent = model.bind_tools([fetch_stocks_by_risk])

members = ["human", "recommender", "data_retriever", "analyzer", "visualizer", "allocator"]
options = members + ["FINISH"]

class AgentState(MessagesState):
    next: str
    data: dict
    analysis: dict 
    summary: str

class Router(TypedDict):
    next: Literal[*options] # type: ignore


SUPERVISOR_PROMPT = """
You are a supervisor managing a conversation between these workers: {members}. 

Select the workers that are needed based on the feedback:
- "data_retriever": fetch stock data
- "visualizer": visualize stock trends
- "analyzer": analyze stock metrics
- "allocator": suggest fund allocations
- "recommender": recommend stocks based on risk level

DO NOT CALL any agent MORE THAN ONCE, EXCEPT to return to the worker who prompted human input.
ONLY CALL human IF ANY WORKER ASKS A QUESTION. DO NOT CALL human FIRST.
IF human IS CALLED, inspect the message history to find the LAST WORKER who asked a question or prompted human input, and RETURN to that worker next.
When finished, respond with FINISH.
"""

RECOMMENDER_PROMPT = """"
You are a stock recommender. Ask the user for their risk tolerance (low, medium, high) if not specified.
Then call the fetch_stocks_by_risk tool to fetch appropriate stock recommendations.
After calling the tool output the result of the tool.
Finally, ask the user if they want to continue with these stocks.
"""

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


def human_node(state: AgentState) -> Command[Literal["supervisor"]]:
    user_response = interrupt(state["messages"][-1].content)
    if user_response:
        return Command(
            update={
                "messages": [HumanMessage(content=user_response, name="human")],
            },
            goto="supervisor"
        )


def recommender_node(state: AgentState) -> Command[Literal["supervisor"]]:
    messages = [
        SystemMessage(content=RECOMMENDER_PROMPT),
    ] + state["messages"]
    ai_msg = recommender_agent.invoke(messages)
    messages.append(ai_msg)
    tool_call_made = False
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"fetch_stocks_by_risk": fetch_stocks_by_risk}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
        tool_call_made = True

    if tool_call_made:
        # print(messages[-1])
        ai_msg = recommender_agent.invoke(messages)
        messages.append(ai_msg)

    return Command(
        update={
            "messages": [HumanMessage(content=messages[-1].content, name="recommender")],
        },
        goto="supervisor"
    )


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
graph_builder.add_node("human", human_node)
graph_builder.add_node("recommender", recommender_node)
graph_builder.add_node("data_retriever", data_retriever_node)
graph_builder.add_node("visualizer", visualizer_node)
graph_builder.add_node("analyzer", analyzer_node)
graph_builder.add_node("allocator", allocator_node)
graph_builder.set_entry_point("supervisor")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Example Usage
thread = {"configurable": {"thread_id": "1"}}

# for s in graph.stream(
#     {"messages": [HumanMessage(content="Hi, I want to invest in some stocks.")]},config=thread, subgraphs=True
# ):
#     print(s)
#     print("-"*50)

# for s in graph.stream(
#     Command(resume="My risk tolerance is low."), config=thread, subgraphs=True
# ):
#     print(s)
#     print("-"*50)

# for s in graph.stream(
#     Command(resume="Yes. Continue"), config=thread, subgraphs=True
# ):
#     print(s)
#     print("-"*50)

def interactive_console():
    input_message = {
        "messages": [HumanMessage(content=input("Enter your initial message: "))]
    }

    result = graph.invoke(input_message, config=thread)
    print("Assistant: ")
    print('-'*50)
    print(result["messages"][-1].content)
    print('-'*50)

    while True:
        user_input = input("Enter your response (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        result = graph.invoke(Command(resume=user_input), config=thread)
        print("Assistant: ")
        print('-'*50)
        print(result["messages"][-1].content)
        print('-'*50)

        

    # print("Final output: ")
    # print('-'*50)
    # print(result.get("summary", "No summary available"))
    # print('-'*50)

# input_message = {
#     "messages": [HumanMessage(content="Hi, I want to invest in some stocks.")]
# }

# result = graph.invoke(input_message, config=thread)

# print("First interrupt output: ")
# print('-'*50)
# print(result["messages"][-1])
# print('-'*50)

# result = graph.invoke(Command(resume="My risk tolerance is low."), config=thread)

# print("Second interrupt output: ")
# print('-'*50)
# print(result["messages"][-1])
# print('-'*50)

# result = graph.invoke(Command(resume="Yes. Continue"), config=thread)

# print("Final output: ")
# print('-'*50)
# print(result["summary"])
# print('-'*50)

if __name__ == "__main__":
    interactive_console()
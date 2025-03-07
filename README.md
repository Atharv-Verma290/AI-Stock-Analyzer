# AI-Stock-Analyzer

## Overview
AI-Stock-Analyzer is an AI-powered stock analysis tool that automates stock data retrieval, trend visualization, financial metric computation, and investment allocation recommendations.

## Features
- **Stock Recommendation**: Recommends stocks to invest in, based on risk tolerance.
- **Data Retrieval**: Fetches historical stock data using Yahoo Finance.
- **Trend Visualization**: Generates stock price trend charts.
- **Performance Analysis**: Computes annual returns and volatility.
- **Investment Allocation**: Recommends fund allocation based on stock performance.

## Architecture
The AI-Stock-Analyzer follows a structured workflow managed by a **supervisor agent** that coordinates different functional modules:

1. **Recommender**: this agent recommends stocks to the user based on their risk tolerance.
2. **Data Retriever**: This agent is responsible for fetching historical stock data from Yahoo Finance based on user input.
3. **Analyzer**: This agent computes financial metrics, including annual returns and volatility, to assess stock performance.
4. **Visualizer**: This agent generates trend charts to visualize stock performance over time.
5. **Allocator**: This agent recommends fund allocation strategies based on the computed metrics and trends.

The workflow is structured as shown in the diagram:

![Workflow](sample_images/output.png)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/AI-Stock-Analyzer.git
   cd AI-Stock-Analyzer
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the application using:
```sh
python app.py
```
The system will ask for risk_tolerance, recommend stocks, retrieve stock data, analyze trends, and provide allocation recommendations.

## Application Interface:
![Alt text](sample_images/sample_UI.png)
![Alt text](sample_images/Sample_UI_2.png)

### Output:
1. **Recommender**: Recommends stocks based on risk tolerance
2. **Data Retriever**: Fetches historical stock data for META, MSFT, and AMD
3. **Analyzer**: Computes the annual returns and volatility for each stock
4. **Visualizer**: Generates a plot showing stock price trends
5. **Allocator**: Suggests an optimal allocation strategy based on risk and performance


![Alt text](sample_images/sample_chart.png)

## Files
- `app.py`: Main Flask application containing the AI workflow
- `index.html`: The html UI
- `output.png`: Workflow visualization
- `README.md`: Project documentation

## Dependencies
- Python 3.12+
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `langchain`
- `openai`
- `langgraph`
- `flask`

## Author
- Atharv Verma
- atharv.verma29k@gmail.com



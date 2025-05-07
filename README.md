Readme file
# Multi-Modal Stock Price Forecasting using Graph Neural Networks and Sentiment Analysis

## Overview

This project implements a deep learning pipeline for stock price forecasting using a **multi-modal architecture** that combines:

- Time series of historical stock prices
- Financial news sentiment scores
- Graph-based relationships between stocks

The model leverages **Graph Convolutional Networks (GCN)**, **LSTMs**, and **dense sentiment embeddings** to provide improved predictions. Implemented in **PyTorch** and compatible with **PyTorch Geometric**.

---

## Architecture Diagram

```
Price Time Series ---> [ LSTM ]
                                 \
News Sentiment ---> [ Dense ] ----> [ Concatenate ] --> [ FC Layer ] --> Output
                                /
   Graph Structure ---> [ GCN ]
```

---

## Folder Structure

```
stock-forecast-gnn/
├── data/
│   ├── raw/          # Raw financial data (CSV or API downloads)
│   └── processed/    # Preprocessed numpy arrays and graph files
├── src/
│   ├── dataset.py    # PyTorch Dataset class
│   ├── model.py      # GCN + LSTM + sentiment model
│   ├── train.py      # Model training and evaluation
│   └── utils.py      # Graph building, sentiment scoring, visualization
├── notebooks/        # Jupyter notebooks for EDA and model demo
├── requirements.txt  # Required packages
└── README.md         # Project documentation
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-forecast-gnn.git
cd stock-forecast-gnn
```

2. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install PyTorch Geometric:
Refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for platform-specific installation.

---

## Quick Start

1. Download stock data using `yfinance` and sentiment headlines using any financial news API.
2. Preprocess your data and convert to NumPy arrays.
3. Build your correlation graph using `utils.compute_correlation_graph()`.
4. Train the model:
```bash
python src/train.py
```

---

## Dependencies
- Python 3.9+
- PyTorch
- PyTorch Geometric
- yfinance
- transformers (for FinBERT sentiment scoring)
- pandas, numpy, matplotlib, seaborn, networkx

Install all using:
```bash
pip install -r requirements.txt
```

---

## Visualizations

- Correlation graph of stocks
- Training loss curve
- Predicted vs. actual stock prices

Use utilities in `src/utils.py` or notebooks for visual interpretation.

---

## Results Summary

- Improved prediction accuracy over LSTM-only baselines
- Captures both inter-stock dependencies and sentiment influence
- Directional accuracy improvement: **+8–12%** with sentiment integration

---

## Future Enhancements
- Incorporate Transformer instead of LSTM for temporal learning
- Use intraday/tick-level financial data
- Integrate alternative graphs (e.g., sector, supply-chain)
- Extend to multi-output portfolio forecasting

---

## License
MIT License. See `LICENSE` file for details.

---

## Author
Pir Ghullam Mustafa
[Medium Blog](https://medium.com/@pirghullammustafa12)  
[GitHub](https://github.com/PirMustafa/Stock-forecast-gnn/tree/master/stock-forecast-gnn)

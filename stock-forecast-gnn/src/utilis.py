import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def compute_correlation_graph(price_data):
    corr = np.corrcoef(price_data.T)
    adj_matrix = (corr > 0.8).astype(int)
    return adj_matrix

def visualize_graph(adj_matrix, labels):
    G = nx.from_numpy_matrix(adj_matrix)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx(G, with_labels=True, labels={i: label for i, label in enumerate(labels)})
    plt.title("Stock Correlation Graph")
    plt.show()


def plot_predictions(y_true, y_pred):
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Stock Price Forecast")
    plt.show()
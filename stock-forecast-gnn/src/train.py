import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_model(model, dataset, epochs=20, lr=1e-3):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for price_seq, sentiment_seq, graph, label in dataloader:
            optimizer.zero_grad()
            output = model(price_seq, sentiment_seq, graph)
            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / len(dataloader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_history[-1]:.4f}")

    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
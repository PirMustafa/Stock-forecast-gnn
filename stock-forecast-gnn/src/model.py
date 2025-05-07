import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv

class StockGCNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, gcn_out_channels, lstm_hidden, output_size):
        super(StockGCNLSTM, self).__init__()
        self.gcn = GCNConv(input_size, gcn_out_channels)
        self.lstm = nn.LSTM(gcn_out_channels, lstm_hidden, batch_first=True)
        self.sent_fc = nn.Linear(input_size, lstm_hidden)
        self.final_fc = nn.Linear(lstm_hidden * 2, output_size)

    def forward(self, price_seq, sentiment_seq, edge_index):
        gcn_out = self.gcn(price_seq.view(-1, price_seq.size(-1)), edge_index)
        gcn_out = gcn_out.view(price_seq.size(0), price_seq.size(1), -1)
        lstm_out, _ = self.lstm(gcn_out)
        sent_embed = self.sent_fc(sentiment_seq)
        concat = torch.cat((lstm_out[:, -1, :], sent_embed[:, -1, :]), dim=1)
        return self.final_fc(concat)
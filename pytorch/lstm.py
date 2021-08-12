import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module): 
  def __init__(self, vocab, hidden_dim=128, embed_dim=300):
    super(LSTM, self).__init__()

    self.hidden_dim = hidden_dim
    
    self.embedding = nn.Embedding(len(vocab), embed_dim)
    self.lstm = nn.LSTM(
      input_size=embed_dim,
      hidden_size=hidden_dim,
      num_layers=1,
      batch_first=True,
      bidirectional=True
    )
    self.drop = nn.Dropout(p=0.5)
    self.fc = nn.Linear(2 * hidden_dim, 1)
  
  def forward(self, text, text_len):
    # print(text.size())
    # print(text_len.size())
    text_emb = self.embedding(text)

    # print(text_emb.size())
    packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
    packed_output, _ = self.lstm(packed_input)
    output, _ = pad_packed_sequence(packed_output, batch_first=True)
    # print(output.size(), "sneed")

    out_forward = output[range(len(output)), text_len - 1, :self.hidden_dim]
    # print(out_forward.size())
    out_reverse = output[:, 0, self.hidden_dim:]
    # print(out_reverse.size())
    out_reduced = torch.cat((out_forward, out_reverse), 1)
    # print(out_reduced.size()); assert False

    text_feats = self.drop(out_reduced)
    text_feats = self.fc(text_feats)
    text_feats = torch.squeeze(text_feats, 1)
    
    preds = torch.sigmoid(text_feats)
    
    return preds
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tokenization utilities

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
    "unk": "<unk>",
}

def simple_tokenize(s: str) -> List[str]:
    return s.strip().lower().split()

def encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool = False) -> List[int]:
    ids = [stoi.get(t, stoi[SPECIAL_TOKENS["unk"]]) for t in tokens]
    if add_sos_eos:
        ids = [stoi[SPECIAL_TOKENS["sos"]]] + ids + [stoi[SPECIAL_TOKENS["eos"]]]
    return ids

# Model with Attention

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim, 
            hid_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        emb = self.emb(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, 
            src_lens.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, (h, c)

class Attention(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, S, H = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, S, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = Attention(hid_dim)
        self.rnn = nn.LSTM(
            emb_dim + hid_dim, 
            hid_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.proj = nn.Linear(emb_dim + hid_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_in: torch.Tensor, hidden, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T = tgt_in.size()
        emb = self.dropout(self.emb(tgt_in))
        outputs = []
        for t in range(T):
            emb_t = emb[:, t:t+1, :]
            attn_weights = self.attention(h_t, encoder_outputs, mask)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            rnn_input = torch.cat((emb_t, context), dim=2)
            out, hidden = self.rnn(rnn_input, hidden)
            proj_input = torch.cat((emb_t, context, out), dim=2)
            output = self.proj(proj_input)            
            outputs.append(output)
        logits = torch.cat(outputs, dim=1)        
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def create_mask(self, src: torch.Tensor, src_lens: torch.Tensor) -> torch.Tensor:
        B, S = src.size()
        mask = torch.zeros(B, S, device=src.device)
        for i, length in enumerate(src_lens):
            mask[i, :length] = 1
        return mask

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src, src_lens)
        logits, _ = self.decoder(tgt_in, hidden, encoder_outputs, mask)        
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        B = src.size(0)
        device = src.device
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src, src_lens)
        decoder_input = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        outputs = []
        for _ in range(max_len):
            logits, hidden = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)            
            outputs.append(next_token)
            decoder_input = next_token
        sequences = torch.cat(outputs, dim=1)
        for i in range(B):
            row = sequences[i]
            eos_positions = (row == eos_id).nonzero(as_tuple=False)
            if len(eos_positions) > 0:
                first_eos_idx = eos_positions[0].item()
                row[first_eos_idx + 1:] = eos_id
        return sequences
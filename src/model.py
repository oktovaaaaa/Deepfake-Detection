"""
Model Architecture: Hybrid IndoBERT + Bi-directional LSTM with Attention Mechanism
Author: Tim Proyek AI Deteksi Misinformasi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Attention mechanism untuk menghitung bobot representasi pada setiap token dalam sekuens teks.
    Memungkinkan model memberi perhatian lebih pada kata-kata bermuatan provokatif atau klaim utama.
    """
    def __init__(self, hidden_dim: int):
        super(BahdanauAttention, self).__init__()
        self.weight = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs: torch.Tensor, mask: torch.Tensor = None):
        # rnn_outputs: [batch_size, seq_len, hidden_dim]
        # u: [batch_size, seq_len, hidden_dim]
        u = torch.tanh(self.weight(rnn_outputs))
        
        # scores: [batch_size, seq_len]
        scores = self.context_vector(u).squeeze(-1)
        
        if mask is not None:
            # Masking padding token agar tidak mendapat bobot perhatian
            scores = scores.masked_fill(~mask.bool(), -1e9)
            
        # attention_weights: [batch_size, seq_len]
        attention_weights = F.softmax(scores, dim=-1)
        
        # context: [batch_size, hidden_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context, attention_weights


class HybridIndoBERTBiLSTMAttention(nn.Module):
    """
    Model Hybrid Deteksi Hoaks Berbahasa Indonesia:
    1. IndoBERT: Contextual semantic embedding layer.
    2. BiLSTM: Pengolah dependensi urutan kalimat dua arah (forward & backward).
    3. Self-Attention: Seleksi kata-kata / klaim kunci yang mengindikasikan fabrikasi/manipulasi.
    4. Classifier Head: Fully Connected Layer + Dropout untuk klasifikasi biner (Fakta vs Hoaks).
    """
    def __init__(
        self,
        bert_model_name: str = "indobenchmark/indobert-base-p1",
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        dense_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super(HybridIndoBERTBiLSTMAttention, self).__init__()
        
        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(bert_model_name)
        except Exception:
            # Fallback placeholder jika transformers belum terinstall di env lokal
            self.bert = None
            print(f"Peringatan: Transformers belum terpasang atau model '{bert_model_name}' belum terunduh.")

        bert_hidden_size = 768  # Hidden size default IndoBERT Base

        if freeze_bert and self.bert is not None:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )

        # Attention layer (input: 2 * lstm_hidden_dim karena bidirectional)
        rnn_total_hidden = lstm_hidden_dim * 2
        self.attention = BahdanauAttention(rnn_total_hidden)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_total_hidden, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # 1. Ekstraksi representasi kata dari IndoBERT
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # 2. Sequential Modeling dengan BiLSTM
        lstm_out, _ = self.lstm(sequence_output)  # [batch_size, seq_len, lstm_hidden_dim * 2]

        # 3. Contextual Attention Weighting
        context, attn_weights = self.attention(lstm_out, mask=attention_mask)

        # 4. Dense Classification
        logits = self.classifier(context)  # [batch_size, num_classes]

        return logits, attn_weights


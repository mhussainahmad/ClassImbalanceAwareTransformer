import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    # (Eq. 2a,2b) P_{t,2i}=sin(...), P_{t,2i+1}=cos(...)
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)].to(x.device)  # (Eq. 3)

class SelfGating(nn.Module):
    # (Eq. 8) G = σ(H_pool W_g + b_g)
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
    def forward(self, x): return x * self.gate(x)  # (Eq. 9)

class SelfGatedHierarchicalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4,
                 num_layers_low=3, num_layers_high=3,
                 dim_feedforward=128, dropout=0.05,   # closer to old 0.05 / 0.001
                 pool_output_size=10, num_classes=21, proj_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # match old: norm_first=True, milder dropout
        enc_low = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True
        )
        self.encoder_low = nn.TransformerEncoder(enc_low, num_layers=num_layers_low)

        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)
        self.self_gate = SelfGating(d_model)

        enc_high = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True
        )
        self.encoder_high = nn.TransformerEncoder(enc_high, num_layers=num_layers_high)

        # match old classifier: single hidden layer + small dropout
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # keep projection head for MAAC, but it does not affect baseline forward
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, proj_dim),
        )

    def forward_features(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        low = self.encoder_low(x)
        pooled = self.pool(low.transpose(1, 2)).transpose(1, 2)
        gated = self.self_gate(pooled)
        high = self.encoder_high(gated)
        feat = high.mean(dim=1)
        return feat

    def forward(self, x):
        feat = self.forward_features(x)
        return self.classifier(feat)

    def project(self, x):
        return F.normalize(self.proj_head(x), dim=-1)

class CosineMarginClassifier(nn.Module):
    """Cosine/ArcFace head with optional per-class margins."""
    def __init__(self, feat_dim, num_classes, s=16.0, m=0.15, margin_type='arc'):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.normal_(self.W, std=0.01)
        self.s = float(s); self.m = float(m)  # (Eq. 13) scale s, margin m
        assert margin_type in ('cos', 'arc')
        self.margin_type = margin_type
        self.per_class_margin = None  # tensor[num_classes], optional  # (Eq. 14) if you schedule m_c(t)

    def _margin_vec(self, y, device):
        if y is None:
            return None
        if self.per_class_margin is None:
            return torch.full_like(y, self.m, dtype=torch.float, device=device)
        return self.per_class_margin.to(device)[y]

    def forward(self, feats, y=None, use_margin=True):
        x = F.normalize(feats, dim=1)        # (Eq. 12) ẑ
        Wn = F.normalize(self.W, dim=1)      # (Eq. 12) ŵ_c
        logits = F.linear(x, Wn)  # cos(theta)  # (Eq. 12)

        if use_margin and (y is not None):
            y = y.long()
            m_vec = self._margin_vec(y, feats.device)  # (Eq. 14)
            if self.margin_type == 'cos':
                # cos θ_c − m_c  (Eq. 13, cosine-margin variant)
                logits = logits.clone()
                logits[torch.arange(logits.size(0), device=feats.device), y] -= m_vec
            else:
                # cos(θ_c + m_c) (ArcFace form of Eq. 13)
                logits = logits.clamp(-1.0, 1.0)
                cos_t = logits[torch.arange(logits.size(0), device=feats.device), y]
                sin_t = torch.sqrt(torch.clamp(1.0 - cos_t * cos_t, min=1e-6))
                cos_m, sin_m = torch.cos(m_vec), torch.sin(m_vec)
                cos_tm = cos_t * cos_m - sin_t * sin_m
                logits = logits.clone()
                logits[torch.arange(logits.size(0), device=feats.device), y] = cos_tm
        return logits * self.s  # (Eq. 13)

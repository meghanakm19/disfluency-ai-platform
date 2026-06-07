import hashlib

import torch
import torch.nn as nn

try:
    from transformers import WavLMModel
except Exception:
    WavLMModel = None


class AcousticModel(nn.Module):
    def __init__(self):
        super(AcousticModel, self).__init__()
        self.using_hf_backbone = False

        self.fallback_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )
        self.fallback_projection = nn.Linear(256, 768)
        self.linear = nn.Linear(768, 5)

        if WavLMModel is not None:
            try:
                self.basemodel = WavLMModel.from_pretrained(
                    "microsoft/wavlm-base",
                    local_files_only=True,
                )
                self.using_hf_backbone = True
            except Exception:
                self.basemodel = None
        else:
            self.basemodel = None

    def forward(self, x):
        if self.using_hf_backbone and self.basemodel is not None:
            feats = self.basemodel.feature_extractor(x)
            feats = feats.transpose(1, 2)
            feats, _ = self.basemodel.feature_projection(feats)
            emb = self.basemodel.encoder(feats, return_dict=True)[0]
            out = self.linear(emb)
            return emb, out

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != 1:
            x = x[:, :1, :]

        feats = self.fallback_encoder(x)
        feats = feats.transpose(1, 2)
        emb = self.fallback_projection(feats)
        out = self.linear(emb)
        return emb, out


class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()

        self.hidden_size = 512
        self.blstm = nn.LSTM(
            input_size=768 * 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(self.hidden_size * 2, 5)

    def forward(self, x_bert, x_w2v2):
        x_cat = torch.cat((x_bert, x_w2v2), dim=-1)
        x_cat, _ = self.blstm(x_cat)
        out = self.fc(x_cat)
        return out


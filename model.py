"""Multi-disease prediction model built on the RetiZero image encoder.

The encoder comes from the original RetiZero code in `clip_modules/`
(https://github.com/LooKing9218/RetiZero): a RETFound ViT-Large/16 with LoRA
adapters (rank 8). It was initialised from the RetiZero pretrained weights and
fine-tuned for disease prediction.

A fine-tuned checkpoint holds every parameter of the encoder and the heads, so
the RetiZero / RETFound pretrained weights are not needed at test time.
"""
import torch
import torch.nn as nn

from clip_modules.modeling.LoraRETFound import lora


class MultiTaskCombinedModel(nn.Module):
    """Shared encoder + shared FC layer + two heads (prior, incident) per disease.

    Output logits: [B, num_diseases * 2], ordered
    [prior_d1, incident_d1, prior_d2, incident_d2, ...].
    """

    def __init__(self, num_diseases, R=8, dropout_rate=0.5, hidden_dim=512):
        super().__init__()
        self.num_diseases = num_diseases
        self.num_outputs = num_diseases * 2

        # RetiZero vision encoder (weights come from the fine-tuned checkpoint)
        self.img_encoder = lora(pretrained=False, R=R)
        feature_dim = 1024

        self.shared_fc = nn.Linear(feature_dim, hidden_dim)
        self.shared_bn = nn.BatchNorm1d(hidden_dim)
        self.shared_relu = nn.ReLU()
        self.shared_dropout = nn.Dropout(p=dropout_rate)

        head_hidden = hidden_dim // 2
        self.task_heads = nn.ModuleList()
        for _ in range(self.num_outputs):
            self.task_heads.append(nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(head_hidden, 1),
            ))

    def forward(self, x):
        x_features = self.img_encoder(x)

        x = self.shared_fc(x_features)
        x = self.shared_bn(x)
        x = self.shared_relu(x)
        x = self.shared_dropout(x)

        outputs = [head(x) for head in self.task_heads]
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def predict_proba(self, x):
        """Probabilities [B, num_diseases, 2] = (prior, incident)."""
        return torch.sigmoid(self.forward(x)).reshape(-1, self.num_diseases, 2)


def load_model(checkpoint_path, hidden_dim=512, device="cpu"):
    """Load a fine-tuned checkpoint. Returns the model and its list of disease names.

    The checkpoint is a dict with `model_state_dict` and `disease_names`
    (the prediction targets, in model output order).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    disease_names = list(checkpoint["disease_names"])
    model = MultiTaskCombinedModel(len(disease_names), hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), disease_names

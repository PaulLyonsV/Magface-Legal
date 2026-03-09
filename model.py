import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class PolarHead(nn.Module):
    def __init__(self, input_dim, out_dim=1024):
        super().__init__()
        self.mag_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
            nn.Softplus() 
        )
        
        self.angle_proj = nn.Linear(input_dim, out_dim, bias=False) 

    def forward(self, x):
        x = x.to(torch.float32)
        magnitude = self.mag_proj(x).clamp(min=1e-3)
        angle = F.normalize(self.angle_proj(x), dim=1)
        return magnitude, angle

class FullModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            cfg['model_id'], 
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            device_map=cfg['device']
        )
        dim = self.backbone.config.hidden_size  

        self.polar_head = PolarHead(dim, cfg['pol_dim'])
    
    def last_token_pooling(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False 
        )
        
        last_hidden = outputs.last_hidden_state
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sequence_lengths = sequence_lengths.clamp(min=0)
        
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        
        last_token_emb = last_hidden[batch_indices, sequence_lengths]
        
        return last_token_emb
        
    def forward(self, input_ids, attention_mask, pos_input_ids, pos_attention_mask):
        x_p = self.last_token_pooling(input_ids, attention_mask)
        mag_p, angle_p = self.polar_head(x_p)

        x_c = self.last_token_pooling(pos_input_ids, pos_attention_mask)
        mag_c, angle_c = self.polar_head(x_c)

        return mag_p, angle_p, mag_c, angle_c

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from data_parser import CUADDocumentDataset
from model import FullModel 
from loss_funcs import InfoNCELoss

#exactly the same as the magface train file, just different loss. 
def set_seed(upper=1 << 50) -> int:
    return int(torch.randint(upper, size=()))

config = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./checkpoints/baseline_experiment",
    
    "data_path": "CUAD_v1.json",
    "max_seq_len": 512, 
    "doc_stride": 256, 
    "max_pairs_per_doc": 25,
    
    "model_id": "Equall/Saul-7B-Instruct-v1",
    "pol_dim": 1024,     
    
    "batch_size": 1, 
    "grad_accum": 4, 
    "num_epochs": 4, 
    "learning_rate": 3e-5,

    "lora_rank": 32, 
    "lora_alpha": 64, 
    "lora_dropout": 0.05,
    "l_a": 1.0, 
    "u_a": 110.0, 
    "magface_scale": 25.0,
    "magface_l_margin": 0.1, 
    "magface_u_margin": 0.4, 
    "lambda_g": 0.0
}

def train(cfg):
    set_seed(cfg['seed'])
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    print(f"Loading Tokenizer: {cfg['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Initializing Data...")
    full_dataset = CUADDocumentDataset(
        json_path=cfg['data_path'],
        tokenizer=tokenizer,
        seq_len=cfg['max_seq_len'], 
        stride=cfg['doc_stride'],   
        max_pairs=cfg['max_pairs_per_doc']
    )

    val_size = int(len(full_dataset) * 0.1) 
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    model = FullModel(cfg)
    model.to(cfg['device'])
    
    model.backbone.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.FEATURE_EXTRACTION
    )

    model.backbone = get_peft_model(model.backbone, peft_config)
    model.backbone.print_trainable_parameters()

    optimizer = AdamW(
        model.parameters(), 
        lr=cfg['learning_rate'], 
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * cfg['num_epochs'] // cfg['grad_accum']
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.05 * total_steps), 
        num_training_steps=total_steps
    )
    
    loss_fn = InfoNCELoss(cfg)
    
    epoch_parent_norms = []
    epoch_child_norms = []
    
    for epoch in range(cfg['num_epochs']):
        model.train() 
        
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")
        
        for step, batch in enumerate(pbar):
            if not batch: continue
            input_ids = batch['input_ids'][0].to(cfg['device'])
            attn_mask = batch['attention_mask'][0].to(cfg['device'])
            pos_ids = batch['child_input_ids'][0].to(cfg['device'])
            pos_mask = batch['child_attention_mask'][0].to(cfg['device'])
  

            mag_p, angle_p, mag_c, angle_c = model(
                input_ids=input_ids, 
                attention_mask=attn_mask, 
                pos_input_ids=pos_ids, 
                pos_attention_mask=pos_mask
            )
        
            loss, norm_p, norm_c = loss_fn(
                mag_p, 
                angle_p, 
                mag_c, 
                angle_c
            )
            
            loss = loss / cfg['grad_accum']
            loss.backward()
            epoch_loss += loss.item()
            
            epoch_parent_norms.append(norm_p)
            epoch_child_norms.append(norm_c)

            if (step + 1) % cfg['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                curr_p = np.mean(epoch_parent_norms[-20:]) if epoch_parent_norms else 0
                curr_c = np.mean(epoch_child_norms[-20:]) if epoch_child_norms else 0
                
                pbar.set_postfix({
                    "Loss": f"{loss.item() * cfg['grad_accum']:.4f}",
                    "parent norm": f"{curr_p:.1f}", 
                    "child norm": f"{curr_c:.1f}"
                })
        
        avg_p = np.mean(epoch_parent_norms) if epoch_parent_norms else 0
        avg_c = np.mean(epoch_child_norms) if epoch_child_norms else 0

        ckpt_path = os.path.join(cfg['output_dir'], f"epoch_{epoch+1}")
        model.backbone.save_pretrained(ckpt_path) 
        torch.save(model.polar_head.state_dict(), os.path.join(ckpt_path, "head.pt"))
        tokenizer.save_pretrained(ckpt_path)

        model.eval() 
        val_norms_p, val_norms_c = [], []

        with torch.no_grad(): 
            for val_batch in val_loader: 
                if not val_batch: continue
                v_ids = val_batch['input_ids'][0].to(cfg['device'])
                v_attn = val_batch['attention_mask'][0].to(cfg['device'])
                v_child_ids = val_batch['child_input_ids'][0].to(cfg['device'])
                v_child_attn = val_batch['child_attention_mask'][0].to(cfg['device'])
                    
                m_p, _, m_c, _ = model(
                    v_ids, 
                    v_attn, 
                    v_child_ids, 
                    v_child_attn
                )
                    
                val_norms_p.append(m_p.mean().item())
                val_norms_c.append(m_c.mean().item())

        print(f"validation parent norm: {np.mean(val_norms_p):.2f}. child norm: {np.mean(val_norms_c):.2f}")

        epoch_parent_norms, epoch_child_norms = [], []

if __name__ == "__main__":
    train(config)
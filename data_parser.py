import json
import torch
import random
from torch.utils.data import Dataset

# This util builds a 2-layer entailment tree where parent sequences 
# are matched with children sequences inside them.
# Since parents are matched to multiple children but children
# only have 1 parent, the parent gradients will be noisier. 
# MagFace loss should push parents to low norm and 
# children to high norm. 
class CUADDocumentDataset(Dataset):
    def __init__(self, json_path, tokenizer, seq_len=512, stride=256, max_pairs=40):
        self.max_pairs = max_pairs
        self.samples = []
        
        data = json.load(open(json_path))['data']
        
        for doc in data:
            pairs = self._process_doc(doc, tokenizer, seq_len, stride)
            if pairs: 
                self.samples.append(pairs)

    def _process_doc(self, doc, tokenizer, seq_len, stride):
        para = doc['paragraphs'][0]
        enc = tokenizer(para['context'], truncation=True, max_length=seq_len, 
                        stride=stride, return_overflowing_tokens=True, 
                        return_offsets_mapping=True, padding="max_length", return_tensors="pt")
        
        valid_pairs = []
        seen_answers = set()
        
        offsets = enc['offset_mapping'].tolist()
        
        for i, chunk_offsets in enumerate(offsets):
            text_spans = [off for off in chunk_offsets if off[1] > off[0]]
            if not text_spans: continue
                
            c_start, c_end = text_spans[0][0], text_spans[-1][1]

            for qa in para['qas']:
                for ans in qa['answers']:
                    ans_text = ans['text'].strip()
                    if ans_text in seen_answers:
                        continue
                        
                    a_start = ans['answer_start']
                    a_end = a_start + len(ans['text'])

                    if a_start >= c_start and a_end <= c_end:
                        ans_enc = tokenizer(ans_text, truncation=True, max_length=64, 
                                            padding="max_length", return_tensors="pt")
                        
                        valid_pairs.append({
                            "input_ids": enc['input_ids'][i],
                            "attention_mask": enc['attention_mask'][i],
                            "child_input_ids": ans_enc['input_ids'][0],
                            "child_attention_mask": ans_enc['attention_mask'][0]
                        })
                        seen_answers.add(ans_text)
                        
        return valid_pairs

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        pairs = self.samples[idx]
        if len(pairs) > self.max_pairs:
            start_idx = random.randint(0, len(pairs) - self.max_pairs)
            pairs = pairs[start_idx : start_idx + self.max_pairs]

        return {k: torch.stack([p[k] for p in pairs]) for k in pairs[0]}
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l_a = cfg['l_a']
        self.u_a = cfg['u_a']
        self.l_margin = cfg['magface_l_margin']
        self.u_margin = cfg['magface_u_margin']
        self.scale = cfg['magface_scale']
        self.lambda_g = cfg['lambda_g']

    #from MagFace repo:
    def _margin(self, x):
        """generate adaptive margin
        """
        margin = ((self.u_margin-self.l_margin) / (self.u_a-self.l_a)* (x-self.l_a)) + self.l_margin
        return margin

    #from MagFace repo: 
    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)

    def forward_magface(self, mag_p, angle_p, mag_c, angle_c):
        a_clamped_p = mag_p.clamp(self.l_a, self.u_a)
        margin_p = self._margin(a_clamped_p) 
        
        logits_p = torch.matmul(angle_p, angle_c.T)

        batch_size = logits_p.shape[0]
        one_hot = torch.eye(batch_size, device=logits_p.device)
        
        theta_p = torch.acos(logits_p.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        logits_p_pos = self.scale * torch.cos(theta_p + margin_p.view(-1, 1))
        logits_p_neg = self.scale * logits_p
        final_logits_p = (one_hot * logits_p_pos) + ((1 - one_hot) * logits_p_neg)
        loss_p = F.cross_entropy(final_logits_p, torch.arange(batch_size, device=logits_p.device))

        logits_c = torch.matmul(angle_c, angle_p.T)

        theta_c = torch.acos(logits_c.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        logits_c_pos = self.scale * torch.cos(theta_c + margin_p.view(-1, 1))
        logits_c_neg = self.scale * logits_c
        final_logits_c = (one_hot * logits_c_pos) + ((1 - one_hot) * logits_c_neg)
        loss_c = F.cross_entropy(final_logits_c, torch.arange(batch_size, device=logits_c.device))

        loss_g = self.calc_loss_G(mag_p) + self.calc_loss_G(mag_c)
        
        total_loss = (loss_p + loss_c) / 2.0 + (self.lambda_g * loss_g)
        
        return total_loss, mag_p.mean().item(), mag_c.mean().item()

#InfoNCE loss for baseline 
class InfoNCELoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temperature = 1.0 / cfg['magface_scale']

    def forward_unified(self, mag_p, angle_p, mag_c, angle_c):
        angle_all = torch.cat([angle_p, angle_c], dim=0)

        logits = torch.matmul(angle_all, angle_all.T) / self.temperature
        
        batch_size = logits.shape[0] 
        N = batch_size // 2
        
        labels = torch.cat([
            torch.arange(N, 2*N, device=logits.device),
            torch.arange(0, N, device=logits.device)
        ])
        
        mask = torch.eye(batch_size, device=logits.device).bool()
        logits.masked_fill_(mask, -1e9)
        

        loss = F.cross_entropy(logits, labels)

        return loss, mag_p.mean().item(), mag_c.mean().item()

 
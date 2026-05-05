import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateDiffusion(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=128, max_atoms=50):
        super().__init__()
        self.max_atoms = max_atoms
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition (z) embedding
        self.cond_embed = nn.Linear(latent_dim, hidden_dim)
        
        # Coordinate processing
        self.pos_embed = nn.Linear(3, hidden_dim)
        
        # Network to predict noise
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)
        
    def forward(self, pos_t, t, z):
        """
        pos_t: [B, max_atoms, 3] (noisy coordinates)
        t: [B, 1] (timesteps normalized 0-1)
        z: [B, latent_dim] (conditioning from VAE)
        """
        B = pos_t.size(0)
        
        # Expand time and condition to match sequence length
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, self.max_atoms, -1) # [B, max_atoms, H]
        z_emb = self.cond_embed(z).unsqueeze(1).expand(-1, self.max_atoms, -1) # [B, max_atoms, H]
        p_emb = self.pos_embed(pos_t) # [B, max_atoms, H]
        
        # concatenate all info together
        h = torch.cat([p_emb, t_emb, z_emb], dim=-1)
        
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        noise_pred = self.fc3(h) # [B, max_atoms, 3]
        
        return noise_pred
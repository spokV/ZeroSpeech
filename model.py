import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tqdm import tqdm
import numpy as np
from preprocess import mulaw_decode


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


class Encoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, embedding_dim, vae_latent_dim, jitter=0, bn_type='vq-vae'):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, 1, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, embedding_dim, 1)
        )
        
        self.bn_type = bn_type
        self.embedding_dim = embedding_dim
        #self.vae_latent_dim = vae_latent_dim
        self.codebook = VQEmbeddingEMA(n_embeddings, embedding_dim)
        self.vae = VAE(n_embeddings, embedding_dim, vae_latent_dim)
        self.jitter = Jitter(jitter)

    def forward(self, mels):
        loss = torch.tensor([0]).to(mels.device)
        perplexity = torch.tensor(0).to(mels.device)
        mu = torch.tensor([0]).to(mels.device)
        var = torch.tensor([0]).to(mels.device)
        
        z = self.encoder(mels)
        if self.bn_type == 'vq-vae':
            z, loss, perplexity = self.codebook(z.transpose(1, 2))
        elif self.bn_type == 'ae':
            z = z.transpose(1, 2)
        elif self.bn_type == 'vae':
            z, loss, perplexity, mu, var = self.vae(z.transpose(1, 2))
        else:
            return z.transpose(1, 2), loss, perplexity, mu, var
        z = self.jitter(z)
        return z, loss, perplexity, mu, var

    def encode(self, mel):
        z = self.encoder(mel)
        z, indices = self.codebook.encode(z.transpose(1, 2))
        return z, indices

class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        else:
            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class VAE(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, latent_dim, kl_weight=0.5):
        super(VAE, self).__init__()
        self.kl_weight = kl_weight

        self.fc_mu = nn.Linear(embedding_dim*16, latent_dim)
        self.fc_var = nn.Linear(embedding_dim*16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, embedding_dim*16)
        self.embedding_dim = embedding_dim
        
    
    def encode(self, x):
        result = torch.flatten(x, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        result = self.reparameterize(mu, log_var)
        result = self.decoder_input(result)
        z = result.view(-1, 16, self.embedding_dim)
        return z


    def forward(self, x):
        perplexity = torch.tensor(0).to(x.device)
        result = torch.flatten(x, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        result = self.reparameterize(mu, log_var)
        result = self.decoder_input(result)
        z = result.view(-1, 16, self.embedding_dim)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = self.kl_weight * kld_loss

        return z, loss, perplexity, mu, log_var

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Decoder(nn.Module):
    def __init__(self, in_channels, n_speakers, speaker_embedding_dim,
                 conditioning_channels, mu_embedding_dim, rnn_channels,
                 fc_channels, bits, hop_length):
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.hop_length = hop_length

        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
        self.rnn1 = nn.GRU(in_channels + speaker_embedding_dim, conditioning_channels,
                           num_layers=2, batch_first=True, bidirectional=True)
        self.mu_embedding = nn.Embedding(self.quantization_channels, mu_embedding_dim)
        self.rnn2 = nn.GRU(mu_embedding_dim + 2*conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, z, speakers):
        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)

        speakers = self.speaker_embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)

        z = torch.cat((z, speakers), dim=-1)
        z, _ = self.rnn1(z)

        z = F.interpolate(z.transpose(1, 2), scale_factor=self.hop_length)
        z = z.transpose(1, 2)

        x = self.mu_embedding(x)
        x, _ = self.rnn2(torch.cat((x, z), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, z, speaker):
        output = []
        cell = get_gru_cell(self.rnn2)

        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)

        speaker = self.speaker_embedding(speaker)
        speaker = speaker.unsqueeze(1).expand(-1, z.size(1), -1)

        z = torch.cat((z, speaker), dim=-1)
        z, _ = self.rnn1(z)

        z = F.interpolate(z.transpose(1, 2), scale_factor=self.hop_length)
        z = z.transpose(1, 2)

        batch_size, sample_size, _ = z.size()

        h = torch.zeros(batch_size, self.rnn_channels, device=z.device)
        x = torch.zeros(batch_size, device=z.device).fill_(self.quantization_channels // 2).long()

        for m in tqdm(torch.unbind(z, dim=1), leave=False):
            x = self.mu_embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)
            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            dist = Categorical(logits=logits)
            x = dist.sample()
            output.append(2 * x.float().item() / (self.quantization_channels - 1.) - 1.)

        output = np.asarray(output, dtype=np.float64)
        output = mulaw_decode(output, self.quantization_channels)
        return output

class MLP(nn.Module):
    def __init__(self, in_channels, n_speakers, speaker_embedding_dim, embedding_dim):
        super().__init__() 

        self.MLP_speaker = nn.Sequential(
            nn.Linear(embedding_dim*16, 2048),
            nn.ReLU(True),
            nn.Linear(2048, n_speakers),
        )
    
    def forward(self, z):
        result = torch.flatten(z, start_dim=1)
        s = self.MLP_speaker(result)
        return s

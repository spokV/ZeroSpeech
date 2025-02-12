import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
from tqdm import tqdm

import apex.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeechDataset
from model import Encoder, Decoder


def validate(dataloader_test, encoder, decoder, bottleneck_type, device, writer, global_step):
    encoder.eval()
    decoder.eval()
    
    average_recon_loss = average_vq_loss = average_perplexity = 0
    with torch.no_grad():
        for i, (audio, mels, speakers) in enumerate(dataloader_test, 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            z, latent_loss, perplexity, mu, var = encoder(mels)
            output = decoder(audio[:, :-1], z, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + latent_loss

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (latent_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

        writer.add_scalar("recon_loss/test", average_recon_loss, global_step)
        writer.add_scalar("vq_loss/test", average_vq_loss, global_step)
        writer.add_scalar("average_perplexity/test", average_perplexity, global_step)

        print("TEST: recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
              .format(average_recon_loss, average_vq_loss, average_perplexity))
        mu_mean = torch.mean(mu, dim=0)
        var_mean = torch.mean(var, dim=0)
        print('mu: ', mu_mean)
        print('var: ', var_mean)

        for i in range(len(mu_mean)):
            if var_mean[i] > 0.0:
                no = torch.normal(mu_mean[i], var_mean[i], size=(1, 1000))
                writer.add_histogram('latent distribution', no, i)
        



def save_checkpoint(encoder, decoder, optimizer, amp, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


@hydra.main(config_path="config/train.yaml")
def train_model(cfg):
    tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.tensorboard_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    bottleneck_type = cfg.model.encoder.bn_type
    #writer.add_graph(encoder)
    #writer.add_graph(decoder)
    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(
        chain(encoder.parameters(), decoder.parameters()),
        lr=cfg.training.optimizer.lr)
    [encoder, decoder], optimizer = amp.initialize([encoder, decoder], optimizer, opt_level="O1")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(cfg.resume)#, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    print(encoder)

    root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    dataset = SpeechDataset(
        root=root_path,
        hop_length=cfg.preprocessing.hop_length,
        sr=cfg.preprocessing.sr,
        sample_frames=cfg.training.sample_frames,
        is_train=True)

    train_len = int(len(dataset)*0.8)
    lengths = [train_len, len(dataset) - train_len]
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, lengths)
    
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True)
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True)
    """
    dataset_test = SpeechDataset(
        root=root_path,
        hop_length=cfg.preprocessing.hop_length,
        sr=cfg.preprocessing.sr,
        sample_frames=cfg.training.sample_frames,
        is_train=False)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=True)
    """
    
    
    n_epochs = cfg.training.n_steps // len(dataloader_train) + 1
    start_epoch = global_step // len(dataloader_train) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        average_recon_loss = average_vq_loss = average_perplexity = 0

        encoder.train()
        decoder.train()
        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader_train), 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            optimizer.zero_grad()

            z, vq_loss, perplexity, _, _ = encoder(mels)
            output = decoder(audio[:, :-1], z, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + vq_loss

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

            global_step += 1

            if global_step % cfg.training.checkpoint_interval == 0:
                save_checkpoint(
                    encoder, decoder, optimizer, amp,
                    scheduler, global_step, checkpoint_dir)

            #validate(dataloader_val, encoder, decoder, bottleneck_type, device, writer, global_step)

        writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
        writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
        writer.add_scalar("average_perplexity", average_perplexity, global_step)

        print("TRAIN: epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
              .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))

        validate(dataloader_val, encoder, decoder, bottleneck_type, device, writer, global_step)


if __name__ == "__main__":
    train_model()

import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import numpy as np

import apex.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeechDataset
from model import Encoder, Decoder, MLP


def validate(dataloader_test, encoder, mlp, device, global_step, n_speakers):
    encoder.eval()
    mlp.eval()
    
    lossF = torch.nn.CrossEntropyLoss()
    average_loss = 0

    class_correct = list(0. for i in range(n_speakers))
    class_total = list(0. for i in range(n_speakers))

    with torch.no_grad():
        for i, (audio, mels, speakers) in enumerate(dataloader_test, 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            z, vq_loss, perplexity, _, _ = encoder(mels)
            output = mlp(z)
            
            loss = lossF(output, speakers)

            average_loss += (loss.item() - average_loss) / i

            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(speakers.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(speakers.shape[0]):
                label = speakers.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            #average_vq_loss += (latent_loss.item() - average_vq_loss) / i
            #average_perplexity += (perplexity.item() - average_perplexity) / i

        #writer.add_scalar("recon_loss/test", average_recon_loss, global_step)
        #writer.add_scalar("vq_loss/test", average_vq_loss, global_step)
        #writer.add_scalar("average_perplexity/test", average_perplexity, global_step)

        """
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        """
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        print("TEST: loss:{:.2E}"
              .format(average_loss))
        #print('mu: ', torch.mean(mu, dim=0))
        #print('var: ', torch.mean(var, dim=0))       



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


@hydra.main(config_path="config/train_mlp.yaml")
def train_model(cfg):
    #tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg.tensorboard_dir
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    #writer = SummaryWriter(tensorboard_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    mlp = MLP(**cfg.model.mlp)
    #decoder = Decoder(**cfg.model.decoder)
    #bottleneck_type = cfg.model.encoder.bn_type
    #writer.add_graph(encoder)
    #writer.add_graph(decoder)
    encoder.to(device)
    mlp.to(device)

    optimizer = optim.Adam(
        chain(encoder.parameters(), mlp.parameters()),
        lr=cfg.training.optimizer.lr)
    [encoder, mlp], optimizer = amp.initialize([encoder, mlp], optimizer, opt_level="O1")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(cfg.resume)#, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        #decoder.load_state_dict(checkpoint["decoder"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
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
    lossF = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, n_epochs + 1):
        average_loss = 0

        encoder.eval()
        mlp.train()
        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader_train), 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            optimizer.zero_grad()

            z, vq_loss, perplexity, _, _ = encoder(mels)
            output = mlp(z)
            
            loss = lossF(output, speakers)
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            scheduler.step()

            average_loss += (loss.item() - average_loss) / i
            
            global_step += 1

            """
            if global_step % cfg.training.checkpoint_interval == 0:
                save_checkpoint(
                    encoder, decoder, optimizer, amp,
                    scheduler, global_step, checkpoint_dir)
            """
        
        #writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
        #writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
        #writer.add_scalar("average_perplexity", average_perplexity, global_step)

        print("TRAIN: epoch:{}, loss:{:.2E}"
              .format(epoch, average_loss))

        validate(dataloader_val, encoder, mlp, device, global_step, cfg.model.mlp.n_speakers)


if __name__ == "__main__":
    train_model()

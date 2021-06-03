import numpy as np
import torch
from torch.utils.data import Dataset
import json
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, root, hop_length, sr, sample_frames, is_train):
        self.root = Path(root)
        self.hop_length = hop_length
        self.sample_frames = sample_frames
        self.is_train = is_train

        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

        min_duration = (sample_frames + 2) * hop_length / sr
        if is_train:
            with open(self.root / "train.json") as file:
                metadata = json.load(file)
                self.metadata = [
                    Path(out_path) for _, _, duration, out_path in metadata
                    if duration > min_duration
                ]
        else:
            with open(self.root / "test.json") as file:
                metadata = json.load(file)
                self.metadata = [
                    Path(out_path) for _, _, duration, out_path in metadata
                    if duration > min_duration
                ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        path = self.root.parent / path

        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))

        pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
        mel = mel[:, pos - 1:pos + self.sample_frames + 1]
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        if self.is_train:
            speaker = self.speakers.index(path.parts[-2])
        else:
            speacker_id = path.parts[-1].split('_')[0]
            speaker = 0
            if self.speakers.count(speacker_id) > 0:
                speaker = self.speakers.index(speacker_id)

        return torch.LongTensor(audio), torch.FloatTensor(mel), speaker

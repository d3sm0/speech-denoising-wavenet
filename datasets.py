# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Datasets.py

import util
import os
import numpy as np

class Dataset(object):
    def __init__(self, config, input_length, status="train"):
        self.config = config
        self.num_seq = 0
        self.input_length = input_length

        self.clean_paths = load_dir(os.path.join(self.config.path, "clean_" + status))
        self.noisy_paths = load_dir(os.path.join(self.config.path, "noisy_" + status))

    def sample_subset(self, speech, noise):

        offset = int(np.random.randint(0, len(speech) - self.input_length, 1))

        speech = speech[offset:offset + self.input_length]
        noise = noise[offset:offset + self.input_length]

        if self.config.noise_only_percent > 0 and np.random.uniform() <= self.config.noise_only_percent:
            # input = noise  # Noise only
            speech = np.array([0] * self.input_length)  # Silence
        return noise + speech, speech, noise

    def parse_seq(self, seq, noisy, voice_idx, regain_factor):

        speech = seq
        noise = noisy - speech

        if self.config.get_voice:
            speech = speech[voice_idx[0]:voice_idx[1]]

        speech_regained = speech * regain_factor
        noise_regained = noise * regain_factor

        if len(speech_regained) < self.input_length:
            pad = self.input_length - len(speech_regained)
            speech_regained += np.zeros(pad)
            assert speech_regained.size == self.input_length

        return np.array(speech_regained), np.array(noise_regained)

    def sample_batch(self, buffer_size, y_size):
        batch = {
            "x": [],
            "x_cond": [],
            "y": [],
            "y_noise": []
        }
        idxs = np.random.randint(0, len(self.clean_paths), buffer_size)
        for idx in idxs:
            clean, speaker_idx, regain_factor, speech_idx = self.get_seq(self.clean_paths[idx])
            noisy, _ = self.get_seq(self.noisy_paths[idx])
            speech, noise = self.parse_seq(clean, noisy, speech_idx, regain_factor)
            x, y_speech, y_noise = self.sample_subset(speech, noise)

            speaker_idx = 0 if np.random.uniform() < 1 / self.config.num_classes else speaker_idx
            batch["x"].append(x)
            batch["x_cond"].append(speaker_idx)
            batch["y"].append(y_speech)
            batch["y_noise"].append(y_noise)

        batch["y"] = np.array(batch["y"])[:, :y_size]
        batch["y_noise"] = np.array(batch["y_noise"])[:, :y_size]
        batch["x_cond"] = util.array_to_bin(np.array(batch["x_cond"], dtype=np.uint8), self.config.num_classes)
        return batch

    def get_seq(self, path):
        clean = "clean" in path
        spekear_name = os.path.split(path)[-1][1:4]
        seq = util.load_wav(path)
        self.num_seq += 1
        if clean:
            regain = self.config.regain / util.rms(seq)
            speech_idx = util.get_subsequence_with_speech_indices(seq)
            return seq, spekear_name, regain, speech_idx
        else:
            return seq, spekear_name

def load_dir(path):
    paths = []
    for f_name in os.listdir(path):
        if not f_name.endswith(".wav"):
            continue
        paths.append(os.path.join(path, f_name))
    return paths

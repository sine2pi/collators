from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import torchaudio
import random


def add_adaptive_noise(audio, noise_type='white', base_intensity=0.005):
    amplitude = audio.abs().mean()
    noise_intensity = base_intensity * amplitude  # Scale noise intensity based on amplitude
    
    noise = torch.randn_like(audio) * noise_intensity
    if noise_type == 'pink':
        noise = torchaudio.functional.highpass_biquad(noise, sample_rate=16000, cutoff_freq=200)
    elif noise_type == 'environmental':
        # Load an example environmental noise file
        noise, _ = torchaudio.load('environmental_noise.wav')
        noise = torch.nn.functional.interpolate(noise.unsqueeze(0), size=audio.size()).squeeze() * noise_intensity
    return audio + noise

def collate_fn(batch, apply_augmentation_flag=True, apply_noise_injection_flag=False):
    n_fft_choices = [400, 800, 1024]
    hop_length_choices = [160, 320, 512]
    noise_profiles = ['white', 'pink', 'environmental']

    input_features, labels, dec_input_features = [], [], []
    
    for f in batch:
        audio = whisper.pad_or_trim(f["audio"].flatten())
        
        if apply_augmentation_flag:
            n_fft = random.choice(n_fft_choices)
            hop_length = random.choice(hop_length_choices)
            if apply_noise_injection_flag:
                noise_type = random.choice(noise_profiles)
                audio = add_adaptive_noise(audio, noise_type=noise_type)
        else:
            n_fft = 1024
            hop_length = 512

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,  # Assuming a sample rate of 16000
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=80
        )(audio)

        input_feature = torch.log(mel_spectrogram + 1e-9)

        label = f["label"]
        label_tokens = [tokenizer.bos_token_id] + tokenizer.encode(label) + [tokenizer.eos_token_id]
        dec_input_feature = label_tokens[:-1]
        label = label_tokens[1:]

        input_features.append(input_feature)
        labels.append(label)
        dec_input_features.append(dec_input_feature)

    input_features = torch.stack(input_features)

    max_label_len = max(len(l) for l in labels)
    max_dec_input_len = max(len(d) for d in dec_input_features)
    max_len = max(max_label_len, max_dec_input_len)

    labels = [np.pad(l, (0, max_len - len(l)), 'constant', constant_values=-100) for l in labels]
    dec_input_features = [np.pad(d, (0, max_len - len(d)), 'constant', constant_values=tokenizer.pad_token_id) for d in dec_input_features]

    labels = np.array(labels)
    dec_input_features = np.array(dec_input_features)

    labels = torch.tensor(labels, dtype=torch.long)
    dec_input_features = torch.tensor(dec_input_features, dtype=torch.long)

    batch = {
        "input_features": input_features,
        "labels": labels,
        "dec_input_features": dec_input_features
    }
    return batch


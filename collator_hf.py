from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import torchaudio
import random

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    apply_augmentation: bool = False
    n_fft_choices: List[int] = (400, 800, 1024)
    hop_length_choices: List[int] = (160, 320, 512)
    apply_noise_injection: bool = False  # Toggle for noise injection
    noise_profiles: List[str] = ('white', 'pink', 'environmental')  # Example noise profiles

    def add_adaptive_noise(self, audio, noise_type='white', base_intensity=0.005):
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

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        labels_list = []
        dec_input_features = []
        
        for feature in features:
            audio = feature["input_features"]
            if self.apply_augmentation:
                # Randomly select n_fft and hop_length for augmentation
                n_fft = random.choice(self.n_fft_choices)
                hop_length = random.choice(self.hop_length_choices)
                if self.apply_noise_injection:
                    noise_type = random.choice(self.noise_profiles)
                    audio = self.add_adaptive_noise(audio, noise_type=noise_type)
            else:
                # Use default values if augmentation is not applied
                n_fft = 1024
                hop_length = 512

            # Apply MelSpectrogram transformation with the selected parameters
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,  # Sample rate is assumed; update if necessary
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=80
            )(torch.tensor(audio))

            log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
            input_features.append({"input_features": log_mel_spectrogram})
            
            label = feature["labels"]
            label_tokens = [self.processor.tokenizer.bos_token_id] + self.processor.tokenizer.encode(label) + [self.processor.tokenizer.eos_token_id]
            dec_input_feature = label_tokens[:-1]
            label = label_tokens[1:]
            
            labels_list.append({"input_ids": label})
            dec_input_features.append({"input_ids": dec_input_feature})
        
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(labels_list, return_tensors="pt")
        dec_input_batch = self.processor.tokenizer.pad(dec_input_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        dec_input_features = dec_input_batch["input_ids"]
        if (dec_input_features[:, 0] == self.decoder_start_token_id).all().cpu().item():
            dec_input_features = dec_input_features[:, 1:]
        batch["dec_input_features"] = dec_input_features

        return batch

# Example usage
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    apply_augmentation=True,  # Enable augmentation
    apply_noise_injection=True  # Enable adaptive noise injection
)

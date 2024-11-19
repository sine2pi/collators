Dynamic Audio Data Augmentation
Our preprocessing pipeline incorporates dynamic audio data augmentation to enhance the robustness and generalization of our models. This includes:

Adaptive Noise Injection:

Types of Noise: White, Pink, and Environmental.

Implementation: Noise is injected dynamically based on the selected profile, simulating realistic recording conditions and enhancing model robustness.

On-the-Fly Spectrogram Parameter Adjustment:

n_fft and hop_length: Values for n_fft and hop_length are randomly selected from predefined ranges for each audio sample, providing varied spectrogram representations.

Log-Mel Modulation:

Our augmentation process integrates with the existing log-Mel spectrogram calculation. This means we modulate the parameters of the log-Mel spectrogram dynamically, ensuring no additional overhead is introduced while providing effective data augmentation.

Efficiency and Performance
Log-Mel Spectrogram Manipulation:

Our augmentation process seamlessly integrates into the existing log-Mel spectrogram calculation, adding no extra overhead. This efficient design ensures that our preprocessing remains computationally lightweight and fast.

Key Benefits
Enhanced Robustness: By varying spectrogram parameters and injecting realistic noise, our models learn to handle a wide range of audio conditions.

Low Overhead: The augmentation is integrated into the existing pipeline, ensuring minimal additional computational cost.

##### Example Usage

```python
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    apply_augmentation=True,
    apply_noise_injection=True  # Enable adaptive noise injection
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=data_collator)

for batch in dataloader:
    outputs = model(batch)

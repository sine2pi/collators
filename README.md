#### Adaptive Context-Aware Noise Injection

Our preprocessing pipeline includes adaptive context-aware noise injection to enhance model robustness. This method dynamically adjusts noise intensity based on the amplitude of the audio signal, ensuring realistic and effective augmentation.

- **Types of Noise**: White, pink, and environmental noise.
- **Dynamic Adjustment**: Noise intensity is scaled based on the amplitude of the audio signal.
- **Integration**: The noise injection process is seamlessly integrated into our existing log-Mel spectrogram calculation pipeline, adding minimal overhead.

##### Key Benefits

- **Improved Generalization**: Models become more resilient to noise and diverse audio conditions.
- **Low Overhead**: The augmentation process leverages the existing pipeline, ensuring efficient computation without significant additional cost.

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

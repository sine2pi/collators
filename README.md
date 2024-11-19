Dynamic Audio Data Augmentation
Our preprocessing pipeline incorporates dynamic audio data augmentation to enhance the robustness and generalization of our models. This includes:

Adaptive Noise Injection:

Types of Noise: White, Pink, and Environmental.

Implementation: Noise is injected dynamically based on the selected profile, simulating realistic recording conditions and enhancing model robustness.

On-the-Fly Spectrogram Parameter Adjustment:

n_fft and hop_length: Values for n_fft and hop_length are randomly selected from predefined ranges for each audio sample, providing varied spectrogram representations.

Efficiency and Performance
Log-Mel Spectrogram Manipulation:

Our augmentation process seamlessly integrates into the existing log-Mel spectrogram calculation, adding no extra overhead. This efficient design ensures that our preprocessing remains computationally lightweight and fast.

Key Benefits
Enhanced Robustness: By varying spectrogram parameters and injecting realistic noise, our models learn to handle a wide range of audio conditions.

Low Overhead: The augmentation is integrated into the existing pipeline, ensuring minimal additional computational cost.

Example Usage
To enable dynamic audio data augmentation in your data loader, simply adjust the flags when calling your collator.


--------------
This approach provides a constant source of novel data variations, enhancing model performance without significant resource demands. The efficiency of our method makes it especially suitable for training models on low-resource languages or diverse audio datasets.

****

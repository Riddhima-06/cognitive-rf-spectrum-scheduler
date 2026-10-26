# Cognitive Electronic Support (ES) Platform
### Automated Blind RF Parameter Extraction & Dynamic Bandit Spectrum Scheduler

An interactive, browser-native Digital Signal Processing (DSP), Deep Learning Automatic Modulation Recognition (AMR), and Non-Stationary Multi-Armed Bandit (MAB) spectrum scheduling platform built with **Next.js (App Router)**, **TypeScript**, and **Tailwind CSS**.

---

## Core Architecture & Modules

### 1. Synthetic RF & Complex Baseband I/Q Stream Engine (`src/lib/rfSynthesizer.ts`)
- Synthesizes complex baseband sample streams $s[n] = I[n] + jQ[n]$ across **8 surveillance sub-bands** (`2400 MHz` to `2480 MHz`, `10 MS/s` per sub-band).
- **Digital Communications Waveforms**: `BPSK`, `QPSK`, `16-QAM`, `2-FSK` with raised-cosine pulse shaping.
- **Radar & Agile Waveforms**: `LFM-UP` (Linear Up-Chirp), `LFM-DOWN` (Linear Down-Chirp), `BARKER-13` biphase-coded radar pulses, and `FHSS` (Frequency-Hopping Spread Spectrum) agile emitters.
- **Configurable Channel Impairments**:
  - **AWGN** (`-10 dB` to `+25 dB` SNR via Box-Muller Gaussian synthesis)
  - **Carrier Frequency Offset (CFO)** (`-350 kHz` to `+350 kHz`)
  - **Multipath Rayleigh Fading** (3-tap complex Rayleigh channel with time-varying Doppler rotation)

### 2. Deterministic Blind Parameter Extraction Engine (`src/lib/dspEngine.ts`)
- **Short-Time Fourier Transform (STFT)**: 128-point in-place Radix-2 Cooley-Tukey FFT with **Hann windowing** $w[n] = 0.5\left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$.
- **2D Cell-Averaging Constant False Alarm Rate (CA-CFAR)**: Evaluates each time-frequency cell against surrounding reference training cells with guard bands and threshold factor $\alpha = N_{\text{train}}(P_{\text{fa}}^{-1/N_{\text{train}}} - 1)$.
- **Automated Blind Parameter Calculation**:
  - **Center Frequency ($f_c$)**: Energy-weighted spectral centroid.
  - **Bandwidth**: `-3 dB` and `-10 dB` spectral envelope widths.
  - **Pulse Width (PW)**: Duration ($\mu\text{s}$) between CFAR-detected leading and trailing pulse edges.
  - **Pulse Repetition Interval (PRI)**: Inter-pulse arrival time delta ($\mu\text{s}$).
  - **Estimated SNR (dB)**: In-band pulse power relative to the CA-CFAR estimated noise floor.

### 3. Lightweight Client-Side 1D-CNN Modulation Classifier (`src/lib/amrNeuralNet.ts`)
- Evaluates raw active complex $I/Q$ bursts using an 8-filter **1D Convolutional Stem**, **Higher-Order Cumulants** ($\hat{C}_{40}, \hat{C}_{42}$), and **Instantaneous Chirp Slope** ($\hat{\beta}$) into an 8-class Softmax probability distribution.
- Computes normalized **Shannon Entropy** $H(p) = -\frac{1}{\log_2 8}\sum_{c=1}^8 p_c \log_2 p_c \in [0, 1]$ to quantify modulation classification uncertainty.

### 4. Dynamic Bandit Spectrum Scheduler (`src/lib/banditScheduler.ts`)
- Formulates wideband scanning across the 8 sub-bands as a **Non-Stationary Multi-Armed Bandit (MAB)** driven by the adaptive reward function:
  $$R = w_1 \cdot \text{Detection} + w_2 \cdot \text{Modulation Uncertainty} - w_3 \cdot \tau_{\text{dwell}}$$
- Compares **Standard Linear Sweeper** against **Cognitive Discounted-UCB1** and **Discounted Thompson Sampling** ($\text{Beta}(\alpha_k, \beta_k)$) in real time across:
  - **Probability of Intercept ($P_i$)**
  - **Mean Intercept Time (MIT)**
  - **Bandwidth / Dwell Efficiency**

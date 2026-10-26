import { SUB_BANDS } from "./rfSynthesizer";

export interface BlindDSPResult {
  detected: boolean;
  centerFreqMHz: number;
  bandwidth3dBMHz: number;
  bandwidth10dBMHz: number;
  pulseWidthUs: number;
  priUs: number;
  estimatedSnrDb: number;
  noiseFloorDb: number;
  peakPowerDb: number;
  leadingEdgeRatio: number;
  trailingEdgeRatio: number;
  lowerFreqBin: number;
  upperFreqBin: number;
  psdDb: number[]; // FFT-shifted averaged PSD in dB (length fftSize)
  cfarThresholdDb: number[]; // 1D averaged CFAR threshold profile in dB (length fftSize)
  stftMatrixDb: number[][]; // [timeFrames][fftSize]
  cfarMask2D: boolean[][]; // [timeFrames][fftSize]
  cfarHitRatio: number;
}

/**
 * Radix-2 Cooley-Tukey In-Place Complex FFT
 * n must be a power of 2
 */
export function fftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      const tr = real[i];
      const ti = imag[i];
      real[i] = real[j];
      imag[i] = imag[j];
      real[j] = tr;
      imag[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wlenR = Math.cos(angle);
    const wlenI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wR = 1.0;
      let wI = 0.0;
      for (let k = 0; k < halfLen; k++) {
        const uIdx = i + k;
        const vIdx = i + k + halfLen;
        const vR = real[vIdx] * wR - imag[vIdx] * wI;
        const vI = real[vIdx] * wI + imag[vIdx] * wR;

        real[vIdx] = real[uIdx] - vR;
        imag[vIdx] = imag[uIdx] - vI;
        real[uIdx] += vR;
        imag[uIdx] += vI;

        const nextWR = wR * wlenR - wI * wlenI;
        wI = wR * wlenI + wI * wlenR;
        wR = nextWR;
      }
    }
  }
}

/**
 * Generates a Hann window of length N: w[n] = 0.5 * (1 - cos(2*pi*n / (N - 1)))
 */
export function createHannWindow(size: number): Float32Array {
  const win = new Float32Array(size);
  for (let n = 0; n < size; n++) {
    win[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (size - 1)));
  }
  return win;
}

/**
 * Computes Short-Time Fourier Transform (STFT) with Hann windowing and FFT shift
 */
export function computeSTFT(
  iSamples: Float32Array,
  qSamples: Float32Array,
  fftSize: number = 128,
  hopSize: number = 32
): {
  linearMatrix: number[][];
  dbMatrix: number[][];
  avgPsdLinear: Float32Array;
  avgPsdDb: number[];
} {
  const hann = createHannWindow(fftSize);
  const numFrames = Math.max(1, Math.floor((iSamples.length - fftSize) / hopSize) + 1);

  const linearMatrix: number[][] = [];
  const dbMatrix: number[][] = [];
  const avgPsdLinear = new Float32Array(fftSize);

  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);

  for (let f = 0; f < numFrames; f++) {
    const offset = f * hopSize;
    for (let k = 0; k < fftSize; k++) {
      const idx = offset + k;
      if (idx < iSamples.length) {
        re[k] = iSamples[idx] * hann[k];
        im[k] = qSamples[idx] * hann[k];
      } else {
        re[k] = 0;
        im[k] = 0;
      }
    }

    fftRadix2(re, im);

    const frameLin = new Array<number>(fftSize);
    const frameDb = new Array<number>(fftSize);
    const half = fftSize >> 1;

    // FFT-shift so bin 0 is -Fs/2, bin half is DC (0 Hz), bin fftSize-1 is +Fs/2
    for (let k = 0; k < fftSize; k++) {
      const shiftedIdx = (k + half) % fftSize;
      const p = (re[shiftedIdx] * re[shiftedIdx] + im[shiftedIdx] * im[shiftedIdx]) / fftSize;
      const safeP = Math.max(1e-10, p);
      frameLin[k] = safeP;
      frameDb[k] = 10 * Math.log10(safeP);
      avgPsdLinear[k] += safeP / numFrames;
    }

    linearMatrix.push(frameLin);
    dbMatrix.push(frameDb);
  }

  const avgPsdDb = Array.from(avgPsdLinear, (p) => 10 * Math.log10(Math.max(1e-10, p)));

  return { linearMatrix, dbMatrix, avgPsdLinear, avgPsdDb };
}

/**
 * 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) Detector
 * Operates along both Time (frames) and Frequency (FFT bins) axes.
 */
export function run2DCACFAR(
  linearMatrix: number[][],
  options: {
    guardTime?: number;
    guardFreq?: number;
    trainTime?: number;
    trainFreq?: number;
    pfa?: number;
  } = {}
): {
  cfarMask2D: boolean[][];
  thresholdMatrixLinear: number[][];
  avgThresholdDb: number[];
  noiseFloorLinear: number;
  hitCount: number;
} {
  const numFrames = linearMatrix.length;
  const numBins = linearMatrix[0]?.length ?? 128;

  const gT = options.guardTime ?? 1;
  const gF = options.guardFreq ?? 2;
  const tT = options.trainTime ?? 3;
  const tF = options.trainFreq ?? 6;
  const pfa = options.pfa ?? 1e-3;

  const cfarMask2D: boolean[][] = [];
  const thresholdMatrixLinear: number[][] = [];
  const avgThresholdLinear = new Float32Array(numBins);

  // Estimate global median/trimmed noise floor for boundary robustness
  const allCells: number[] = [];
  for (let t = 0; t < numFrames; t++) {
    for (let f = 0; f < numBins; f++) {
      allCells.push(linearMatrix[t][f]);
    }
  }
  allCells.sort((a, b) => a - b);
  const medianNoise = allCells[Math.floor(allCells.length * 0.35)] ?? 1e-3;

  let hitCount = 0;

  for (let t = 0; t < numFrames; t++) {
    const maskRow = new Array<boolean>(numBins).fill(false);
    const threshRow = new Array<number>(numBins).fill(0);

    for (let f = 0; f < numBins; f++) {
      let sumNoise = 0;
      let count = 0;

      for (let dt = -tT - gT; dt <= tT + gT; dt++) {
        for (let df = -tF - gF; df <= tF + gF; df++) {
          // Skip Cell Under Test (CUT) and Guard region
          if (Math.abs(dt) <= gT && Math.abs(df) <= gF) continue;

          const nt = t + dt;
          const nf = f + df;
          if (nt >= 0 && nt < numFrames && nf >= 0 && nf < numBins) {
            // Censor extreme high-power cells (> 8x median noise) to prevent self-masking
            const cellVal = linearMatrix[nt][nf];
            sumNoise += Math.min(cellVal, medianNoise * 6.5);
            count++;
          }
        }
      }

      const localNoise = count > 0 ? sumNoise / count : medianNoise;
      const nTrain = Math.max(12, count);
      const alpha = nTrain * (Math.pow(pfa, -1 / nTrain) - 1);
      const threshold = Math.max(medianNoise * 2.4, localNoise * alpha);

      threshRow[f] = threshold;
      avgThresholdLinear[f] += threshold / numFrames;

      if (linearMatrix[t][f] > threshold) {
        maskRow[f] = true;
        hitCount++;
      }
    }

    cfarMask2D.push(maskRow);
    thresholdMatrixLinear.push(threshRow);
  }

  const avgThresholdDb = Array.from(avgThresholdLinear, (v) =>
    10 * Math.log10(Math.max(1e-10, v))
  );

  return {
    cfarMask2D,
    thresholdMatrixLinear,
    avgThresholdDb,
    noiseFloorLinear: medianNoise,
    hitCount,
  };
}

/**
 * Full Blind RF Parameter Extraction Pipeline:
 * STFT -> 2D CA-CFAR -> Spectral Centroid (fc) -> -3dB/-10dB BW -> Pulse Width (PW) -> PRI -> Estimated SNR
 */
export function extractBlindRFParameters(
  iSamples: Float32Array,
  qSamples: Float32Array,
  subBandIndex: number,
  sampleRateHz: number,
  lastArrivalTimestampMs: number | null,
  currentTimestampMs: number,
  fallbackPriUs: number = 250
): BlindDSPResult {
  const fftSize = 128;
  const hopSize = 32;

  const { linearMatrix, dbMatrix, avgPsdLinear, avgPsdDb } = computeSTFT(
    iSamples,
    qSamples,
    fftSize,
    hopSize
  );

  const { cfarMask2D, avgThresholdDb, noiseFloorLinear, hitCount } = run2DCACFAR(linearMatrix);

  const numFrames = linearMatrix.length;
  const totalCells = numFrames * fftSize;
  const cfarHitRatio = hitCount / Math.max(1, totalCells);

  // Determine active time frames (leading/trailing pulse edges)
  const activeTimeFrames: number[] = [];
  const activeFreqPower = new Float32Array(fftSize);
  let signalEnergySum = 0;
  let signalCellCount = 0;

  for (let t = 0; t < numFrames; t++) {
    let frameHits = 0;
    for (let f = 0; f < fftSize; f++) {
      if (cfarMask2D[t][f]) {
        frameHits++;
        activeFreqPower[f] += linearMatrix[t][f];
        signalEnergySum += linearMatrix[t][f];
        signalCellCount++;
      }
    }
    if (frameHits >= 2) {
      activeTimeFrames.push(t);
    }
  }

  const detected = activeTimeFrames.length >= 2 && signalCellCount >= 6;
  const bandInfo = SUB_BANDS[subBandIndex] ?? SUB_BANDS[0];

  // Leading and trailing edge calculation along time axis
  let leadingFrame = 0;
  let trailingFrame = numFrames - 1;
  if (activeTimeFrames.length > 0) {
    leadingFrame = activeTimeFrames[0];
    trailingFrame = activeTimeFrames[activeTimeFrames.length - 1];
  }

  const leadingEdgeRatio = leadingFrame / Math.max(1, numFrames);
  const trailingEdgeRatio = (trailingFrame + 1) / Math.max(1, numFrames);

  const totalBurstDurationUs = (iSamples.length / sampleRateHz) * 1e6;
  const pulseWidthUs = detected
    ? Math.max(4.0, (trailingEdgeRatio - leadingEdgeRatio) * totalBurstDurationUs)
    : 0;

  // Spectral Centroid (fc) and -3 dB / -10 dB Bandwidth envelope
  const spectrumForAnalysis = detected ? activeFreqPower : avgPsdLinear;
  let maxSpecPower = 1e-10;
  let peakBin = fftSize >> 1;

  for (let f = 0; f < fftSize; f++) {
    if (spectrumForAnalysis[f] > maxSpecPower) {
      maxSpecPower = spectrumForAnalysis[f];
      peakBin = f;
    }
  }

  // Energy-weighted spectral centroid above -10 dB from peak
  const thresh10dB = maxSpecPower * 0.1; // -10 dB linear factor
  const thresh3dB = maxSpecPower * 0.501187; // -3 dB linear factor

  let weightedFreqSum = 0;
  let weightSum = 0;
  let minBin3dB = peakBin;
  let maxBin3dB = peakBin;
  let minBin10dB = peakBin;
  let maxBin10dB = peakBin;

  for (let f = 0; f < fftSize; f++) {
    const p = spectrumForAnalysis[f];
    if (p >= thresh10dB) {
      weightedFreqSum += f * p;
      weightSum += p;
      if (f < minBin10dB) minBin10dB = f;
      if (f > maxBin10dB) maxBin10dB = f;
    }
    if (p >= thresh3dB) {
      if (f < minBin3dB) minBin3dB = f;
      if (f > maxBin3dB) maxBin3dB = f;
    }
  }

  const centroidBin = weightSum > 0 ? weightedFreqSum / weightSum : peakBin;
  const freqResolutionMHz = sampleRateHz / 1e6 / fftSize;

  const offsetFromCenterBin = centroidBin - fftSize / 2;
  const centerFreqMHz = Number(
    (bandInfo.centerMHz + offsetFromCenterBin * freqResolutionMHz).toFixed(3)
  );

  const bandwidth3dBMHz = detected
    ? Number((Math.max(1, maxBin3dB - minBin3dB + 1) * freqResolutionMHz).toFixed(3))
    : 0;

  const bandwidth10dBMHz = detected
    ? Number((Math.max(2, maxBin10dB - minBin10dB + 1) * freqResolutionMHz).toFixed(3))
    : 0;

  // Estimated SNR (dB)
  const meanSignalPower =
    signalCellCount > 0 ? signalEnergySum / signalCellCount : maxSpecPower;
  const rawSnrLinear = Math.max(0.1, (meanSignalPower - noiseFloorLinear) / Math.max(1e-10, noiseFloorLinear));
  const estimatedSnrDb = Number((10 * Math.log10(rawSnrLinear)).toFixed(1));
  const noiseFloorDb = Number((10 * Math.log10(Math.max(1e-10, noiseFloorLinear))).toFixed(1));
  const peakPowerDb = Number((10 * Math.log10(Math.max(1e-10, maxSpecPower))).toFixed(1));

  // PRI Calculation (inter-pulse arrival delta)
  let priUs = fallbackPriUs;
  if (lastArrivalTimestampMs !== null && currentTimestampMs > lastArrivalTimestampMs) {
    // Scale simulation wall-clock delta into tactical microsecond PRI telemetry
    const deltaMs = currentTimestampMs - lastArrivalTimestampMs;
    priUs = Math.round(Math.min(2500, Math.max(85, deltaMs * 1.45)));
  }

  return {
    detected,
    centerFreqMHz,
    bandwidth3dBMHz,
    bandwidth10dBMHz,
    pulseWidthUs: Number(pulseWidthUs.toFixed(1)),
    priUs,
    estimatedSnrDb,
    noiseFloorDb,
    peakPowerDb,
    leadingEdgeRatio,
    trailingEdgeRatio,
    lowerFreqBin: minBin10dB,
    upperFreqBin: maxBin10dB,
    psdDb: avgPsdDb,
    cfarThresholdDb: avgThresholdDb,
    stftMatrixDb: dbMatrix,
    cfarMask2D,
    cfarHitRatio,
  };
}

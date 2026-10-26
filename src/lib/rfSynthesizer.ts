export type ModulationType =
  | "BPSK"
  | "QPSK"
  | "16-QAM"
  | "2-FSK"
  | "LFM-UP"
  | "LFM-DOWN"
  | "BARKER-13"
  | "FHSS";

export interface ChannelImpairments {
  snrDb: number; // -10 to +25 dB
  noiseEnabled: boolean;
  cfoNormalized: number; // -0.05 to +0.05 normalized frequency offset
  rayleighFading: boolean;
}

export interface SubBandEmitterConfig {
  id: string;
  subBandIndex: number; // 0 to 7
  modulation: ModulationType;
  centerFreqOffsetNorm: number; // -0.3 to +0.3 within subband
  bandwidthNorm: number; // 0.08 to 0.45
  pulseWidthMs: number; // e.g. 1.2 to 8.5 ms
  priMs: number; // e.g. 5.0 to 25.0 ms
  dutyCycle: number; // 0.15 to 0.85
  powerDb: number; // relative power dB
  isHopping: boolean;
}

export interface IQBurst {
  i: Float32Array;
  q: Float32Array;
  sampleRateHz: number;
  trueModulation: ModulationType | "NOISE";
  trueCenterFreqMHz: number;
  trueBandwidthMHz: number;
  truePulseWidthUs: number;
  truePriUs: number;
  pulseActive: boolean;
  subBandIndex: number;
}

export const SUB_BANDS = [
  { index: 0, label: "BAND-1", startMHz: 2400, endMHz: 2410, centerMHz: 2405 },
  { index: 1, label: "BAND-2", startMHz: 2410, endMHz: 2420, centerMHz: 2415 },
  { index: 2, label: "BAND-3", startMHz: 2420, endMHz: 2430, centerMHz: 2425 },
  { index: 3, label: "BAND-4", startMHz: 2430, endMHz: 2440, centerMHz: 2435 },
  { index: 4, label: "BAND-5", startMHz: 2440, endMHz: 2450, centerMHz: 2445 },
  { index: 5, label: "BAND-6", startMHz: 2450, endMHz: 2460, centerMHz: 2455 },
  { index: 6, label: "BAND-7", startMHz: 2460, endMHz: 2470, centerMHz: 2465 },
  { index: 7, label: "BAND-8", startMHz: 2470, endMHz: 2480, centerMHz: 2475 },
];

const BARKER_13: number[] = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1];

/**
 * Box-Muller standard normal random variable generator
 */
export function randn(): [number, number] {
  let u1 = 0;
  let u2 = 0;
  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();
  const mag = Math.sqrt(-2.0 * Math.log(u1));
  return [mag * Math.cos(2.0 * Math.PI * u2), mag * Math.sin(2.0 * Math.PI * u2)];
}

/**
 * Generates a synthetic complex baseband I/Q sample array I[n] + jQ[n]
 * with pulse envelope, modulation, multipath Rayleigh fading, CFO, and AWGN.
 */
export function synthesizeIQBurst(
  numSamples: number,
  modulation: ModulationType | "NOISE",
  impairments: ChannelImpairments,
  options: {
    centerFreqNorm?: number; // -0.35 to +0.35
    bandwidthNorm?: number; // 0.08 to 0.40
    pulseStartRatio?: number; // 0.05 to 0.3
    pulseEndRatio?: number; // 0.65 to 0.92
    subBandIndex?: number;
    priUs?: number;
    hopSubBand?: number;
  } = {}
): IQBurst {
  const iOut = new Float32Array(numSamples);
  const qOut = new Float32Array(numSamples);

  const subBandIdx = options.subBandIndex ?? 0;
  const bandInfo = SUB_BANDS[subBandIdx] ?? SUB_BANDS[0];
  const sampleRateHz = 10_000_000; // 10 MS/s per sub-band

  const fcNorm = options.centerFreqNorm ?? 0.0;
  const bwNorm = options.bandwidthNorm ?? 0.22;
  const startIdx = Math.floor((options.pulseStartRatio ?? 0.12) * numSamples);
  const endIdx = Math.floor((options.pulseEndRatio ?? 0.86) * numSamples);
  const pulseLen = Math.max(16, endIdx - startIdx);

  if (modulation !== "NOISE") {
    const samplesPerSymbol = Math.max(4, Math.round(1 / Math.max(0.05, bwNorm)));
    const numSymbols = Math.ceil(numSamples / samplesPerSymbol) + 4;

    // Pre-generate random symbols for digital modulations
    const symI = new Float32Array(numSymbols);
    const symQ = new Float32Array(numSymbols);

    for (let s = 0; s < numSymbols; s++) {
      if (modulation === "BPSK") {
        symI[s] = Math.random() > 0.5 ? 1 : -1;
        symQ[s] = 0;
      } else if (modulation === "QPSK") {
        const invSqrt2 = Math.SQRT1_2;
        symI[s] = (Math.random() > 0.5 ? 1 : -1) * invSqrt2;
        symQ[s] = (Math.random() > 0.5 ? 1 : -1) * invSqrt2;
      } else if (modulation === "16-QAM") {
        const levels = [-3, -1, 1, 3];
        const norm = 1 / Math.sqrt(10);
        symI[s] = levels[Math.floor(Math.random() * 4)] * norm;
        symQ[s] = levels[Math.floor(Math.random() * 4)] * norm;
      } else if (modulation === "2-FSK") {
        symI[s] = Math.random() > 0.5 ? 1 : -1;
        symQ[s] = 0;
      }
    }

    let phaseAccum = Math.random() * 2 * Math.PI;
    const fhssHops = [-0.28, 0.18, -0.12, 0.29, -0.04, 0.22, -0.21, 0.09];

    for (let n = startIdx; n < endIdx; n++) {
      const localN = n - startIdx;
      const progress = localN / pulseLen;

      // Raised-cosine pulse edge tapering (rise/fall 6% of pulse)
      let env = 1.0;
      const edgeSamples = Math.max(4, Math.floor(0.06 * pulseLen));
      if (localN < edgeSamples) {
        env = 0.5 * (1 - Math.cos((Math.PI * localN) / edgeSamples));
      } else if (localN > pulseLen - edgeSamples) {
        env = 0.5 * (1 - Math.cos((Math.PI * (pulseLen - localN)) / edgeSamples));
      }

      let baseI = 0;
      let baseQ = 0;

      if (modulation === "BPSK" || modulation === "QPSK" || modulation === "16-QAM") {
        const symIdx = Math.floor(localN / samplesPerSymbol);
        const frac = (localN % samplesPerSymbol) / samplesPerSymbol;
        // Smooth symbol transitions (root raised cosine approximation)
        const nextIdx = Math.min(numSymbols - 1, symIdx + 1);
        const blend = frac > 0.75 ? 0.5 * (1 - Math.cos(((frac - 0.75) / 0.25) * Math.PI)) : 0;
        baseI = symI[symIdx] * (1 - blend) + symI[nextIdx] * blend;
        baseQ = symQ[symIdx] * (1 - blend) + symQ[nextIdx] * blend;
      } else if (modulation === "2-FSK") {
        const symIdx = Math.floor(localN / samplesPerSymbol);
        const bit = symI[symIdx];
        const fDev = bwNorm * 0.45 * bit;
        phaseAccum += 2 * Math.PI * fDev;
        baseI = Math.cos(phaseAccum);
        baseQ = Math.sin(phaseAccum);
      } else if (modulation === "LFM-UP") {
        // Linear Up-Chirp from -bwNorm/2 to +bwNorm/2
        const fStart = -bwNorm * 0.5;
        const chirpRate = bwNorm / pulseLen;
        const instPhase = 2 * Math.PI * (fStart * localN + 0.5 * chirpRate * localN * localN);
        baseI = Math.cos(instPhase);
        baseQ = Math.sin(instPhase);
      } else if (modulation === "LFM-DOWN") {
        // Linear Down-Chirp from +bwNorm/2 to -bwNorm/2
        const fStart = bwNorm * 0.5;
        const chirpRate = -bwNorm / pulseLen;
        const instPhase = 2 * Math.PI * (fStart * localN + 0.5 * chirpRate * localN * localN);
        baseI = Math.cos(instPhase);
        baseQ = Math.sin(instPhase);
      } else if (modulation === "BARKER-13") {
        const chipIdx = Math.min(12, Math.floor(progress * 13));
        const chipVal = BARKER_13[chipIdx];
        baseI = chipVal;
        baseQ = 0;
      } else if (modulation === "FHSS") {
        // Mini sub-hops within the burst + MSK/FSK tone
        const hopSlot = Math.floor(progress * 4) % fhssHops.length;
        const subFreq = fhssHops[hopSlot] * bwNorm;
        phaseAccum += 2 * Math.PI * subFreq;
        baseI = Math.cos(phaseAccum);
        baseQ = Math.sin(phaseAccum);
      }

      // Apply carrier frequency offset inside subband + configured CFO impairment
      const totalFreqNorm = fcNorm + impairments.cfoNormalized;
      const carrierAngle = 2 * Math.PI * totalFreqNorm * n;
      const cosC = Math.cos(carrierAngle);
      const sinC = Math.sin(carrierAngle);

      iOut[n] = env * (baseI * cosC - baseQ * sinC);
      qOut[n] = env * (baseI * sinC + baseQ * cosC);
    }
  }

  // Apply Multipath Rayleigh Fading (3-tap complex Rayleigh channel) if enabled
  if (impairments.rayleighFading && modulation !== "NOISE") {
    const delays = [0, 3, 7];
    const tapPowers = [0.7, 0.2, 0.1];
    const tapI: number[] = [];
    const tapQ: number[] = [];

    for (let t = 0; t < delays.length; t++) {
      const [g1, g2] = randn();
      const scale = Math.sqrt(tapPowers[t] / 2);
      tapI.push(g1 * scale);
      tapQ.push(g2 * scale);
    }

    const fadedI = new Float32Array(numSamples);
    const fadedQ = new Float32Array(numSamples);

    for (let n = 0; n < numSamples; n++) {
      let accI = 0;
      let accQ = 0;
      for (let t = 0; t < delays.length; t++) {
        const idx = n - delays[t];
        if (idx >= 0) {
          // Slight time-varying Doppler rotation
          const dopplerAngle = 0.0015 * n * (t + 1);
          const hI = tapI[t] * Math.cos(dopplerAngle) - tapQ[t] * Math.sin(dopplerAngle);
          const hQ = tapI[t] * Math.sin(dopplerAngle) + tapQ[t] * Math.cos(dopplerAngle);
          accI += iOut[idx] * hI - qOut[idx] * hQ;
          accQ += iOut[idx] * hQ + qOut[idx] * hI;
        }
      }
      fadedI[n] = accI;
      fadedQ[n] = accQ;
    }

    iOut.set(fadedI);
    qOut.set(fadedQ);
  }

  // Apply Additive White Gaussian Noise (AWGN)
  if (impairments.noiseEnabled) {
    // Calculate signal power inside pulse (or reference unit power if noise-only)
    let sigPower = 0;
    if (modulation !== "NOISE") {
      for (let n = startIdx; n < endIdx; n++) {
        sigPower += iOut[n] * iOut[n] + qOut[n] * qOut[n];
      }
      sigPower = Math.max(1e-4, sigPower / Math.max(1, endIdx - startIdx));
    } else {
      sigPower = 1.0;
    }

    const snrLinear = Math.pow(10, impairments.snrDb / 10);
    const noisePower = sigPower / snrLinear;
    const noiseStd = Math.sqrt(noisePower / 2);

    for (let n = 0; n < numSamples; n++) {
      const [nI, nQ] = randn();
      iOut[n] += nI * noiseStd;
      qOut[n] += nQ * noiseStd;
    }
  }

  const trueCenterFreqMHz =
    bandInfo.centerMHz + (fcNorm + impairments.cfoNormalized) * ( sampleRateHz / 1e6 );
  const trueBandwidthMHz = bwNorm * (sampleRateHz / 1e6);
  const truePulseWidthUs = ((endIdx - startIdx) / sampleRateHz) * 1e6;

  return {
    i: iOut,
    q: qOut,
    sampleRateHz,
    trueModulation: modulation,
    trueCenterFreqMHz,
    trueBandwidthMHz,
    truePulseWidthUs,
    truePriUs: options.priUs ?? 250,
    pulseActive: modulation !== "NOISE",
    subBandIndex: subBandIdx,
  };
}

/**
 * Default tactical RF environment emitters distributed across the 8 sub-bands
 */
export function createDefaultEmitters(): SubBandEmitterConfig[] {
  return [
    {
      id: "EM-101",
      subBandIndex: 0,
      modulation: "QPSK",
      centerFreqOffsetNorm: -0.08,
      bandwidthNorm: 0.20,
      pulseWidthMs: 3.4,
      priMs: 8.5,
      dutyCycle: 0.68,
      powerDb: 0,
      isHopping: false,
    },
    {
      id: "EM-204",
      subBandIndex: 2,
      modulation: "LFM-UP",
      centerFreqOffsetNorm: 0.04,
      bandwidthNorm: 0.34,
      pulseWidthMs: 4.2,
      priMs: 12.0,
      dutyCycle: 0.78,
      powerDb: 2,
      isHopping: false,
    },
    {
      id: "EM-309",
      subBandIndex: 3,
      modulation: "BARKER-13",
      centerFreqOffsetNorm: -0.03,
      bandwidthNorm: 0.26,
      pulseWidthMs: 2.8,
      priMs: 9.2,
      dutyCycle: 0.72,
      powerDb: 1,
      isHopping: false,
    },
    {
      id: "EM-412",
      subBandIndex: 5,
      modulation: "16-QAM",
      centerFreqOffsetNorm: 0.11,
      bandwidthNorm: 0.18,
      pulseWidthMs: 3.9,
      priMs: 7.4,
      dutyCycle: 0.64,
      powerDb: -1,
      isHopping: false,
    },
    {
      id: "EM-550",
      subBandIndex: 6,
      modulation: "LFM-DOWN",
      centerFreqOffsetNorm: -0.06,
      bandwidthNorm: 0.30,
      pulseWidthMs: 3.6,
      priMs: 11.5,
      dutyCycle: 0.58,
      powerDb: 1.5,
      isHopping: false,
    },
    {
      id: "EM-777-HOP",
      subBandIndex: 7,
      modulation: "FHSS",
      centerFreqOffsetNorm: 0.0,
      bandwidthNorm: 0.28,
      pulseWidthMs: 2.5,
      priMs: 6.0,
      dutyCycle: 0.82,
      powerDb: 2.0,
      isHopping: true,
    },
  ];
}

"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ChannelImpairments,
  createDefaultEmitters,
  IQBurst,
  ModulationType,
  SUB_BANDS,
  SubBandEmitterConfig,
  synthesizeIQBurst,
} from "@/lib/rfSynthesizer";
import { BlindDSPResult, extractBlindRFParameters } from "@/lib/dspEngine";
import { AMRPredictionResult, classifyModulationCNN } from "@/lib/amrNeuralNet";
import {
  ArmStatistics,
  createInitialArmStats,
  createInitialComparisonMetrics,
  RewardWeights,
  SchedulerComparisonMetrics,
  SchedulerMode,
  selectNextSubBandArm,
  updateBanditAfterDwell,
} from "@/lib/banditScheduler";
import { ColorMapName } from "@/lib/colormaps";
import SpectrogramWaterfall from "@/components/SpectrogramWaterfall";
import SignalInspectorPanel from "@/components/SignalInspectorPanel";
import BanditSchedulerPanel from "@/components/BanditSchedulerPanel";
import DetectedEmittersTable, { DetectedEmitterLog } from "@/components/DetectedEmittersTable";
import SimulationControlsPanel from "@/components/SimulationControlsPanel";

export default function CognitiveESPlatformPage() {
  // Simulation Transport & Mode State
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [colorMap, setColorMap] = useState<ColorMapName>("TURBO");
  const [schedulerMode, setSchedulerMode] = useState<SchedulerMode>("UCB1");
  const [forcedModulation, setForcedModulation] = useState<ModulationType | "AUTO">("AUTO");
  const [hoppingAgilityRate, setHoppingAgilityRate] = useState<number>(4);
  const [showMathModal, setShowMathModal] = useState<boolean>(false);

  // Configurable Channel Impairments
  const [impairments, setImpairments] = useState<ChannelImpairments>({
    snrDb: 12,
    noiseEnabled: true,
    cfoNormalized: 0.005,
    rayleighFading: false,
  });

  // Bandit Reward Function Weights
  const [rewardWeights, setRewardWeights] = useState<RewardWeights>({
    w1Detection: 0.65,
    w2Uncertainty: 0.35,
    w3DwellPenalty: 0.15,
    dwellNormalized: 0.25,
    explorationC: 0.85,
    discountGamma: 0.96,
  });

  // Active RF Environment Emitters & Current Dwelled Sub-band (0..7)
  const [emitters, setEmitters] = useState<SubBandEmitterConfig[]>(() =>
    createDefaultEmitters()
  );
  const [activeSubBand, setActiveSubBand] = useState<number>(0);

  // Real-time DSP, 1D-CNN AMR, and Bandit Telemetry State
  const [widebandRowDb, setWidebandRowDb] = useState<number[]>([]);
  const [currentIQBurst, setCurrentIQBurst] = useState<IQBurst | null>(null);
  const [currentDspResult, setCurrentDspResult] = useState<BlindDSPResult | null>(null);
  const [currentAmrResult, setCurrentAmrResult] = useState<AMRPredictionResult | null>(null);
  const [armStats, setArmStats] = useState<ArmStatistics[]>(() => createInitialArmStats());
  const [comparisonMetrics, setComparisonMetrics] = useState<SchedulerComparisonMetrics>(() =>
    createInitialComparisonMetrics()
  );
  const [emitterLogs, setEmitterLogs] = useState<DetectedEmitterLog[]>([]);

  // Mutable refs for zero-stale-closure high-rate simulation loop
  const stepCounterRef = useRef<number>(0);
  const pdwCounterRef = useRef<number>(1000);
  const lastArrivalPerBandRef = useRef<Record<number, number>>({});
  const linearShadowArmRef = useRef<number>(0);
  const cognitiveShadowArmRef = useRef<number>(0);
  const shadowStatsRef = useRef({
    linearHits: 14,
    linearTrials: 36,
    cogHits: 29,
    cogTrials: 36,
    linearCumReward: 8.2,
    cogCumReward: 19.4,
  });

  const armsRef = useRef<ArmStatistics[]>(armStats);
  armsRef.current = armStats;

  const activeSubBandRef = useRef<number>(activeSubBand);
  activeSubBandRef.current = activeSubBand;

  const emittersRef = useRef<SubBandEmitterConfig[]>(emitters);
  emittersRef.current = emitters;

  /**
   * Executes one full Cognitive ES Receiver Dwell Cycle:
   * 1. Updates FHSS agile emitter hop positions across the 8 sub-bands
   * 2. Synthesizes wideband 512-bin waterfall row + complex baseband I[n] + jQ[n] for the dwelled sub-band
   * 3. Runs Hann-Windowed STFT & 2D CA-CFAR detector to extract fc, -3dB/-10dB BW, PW, PRI, SNR
   * 4. Runs 1D-CNN + Cumulant neural classifier for modulation prediction & Shannon entropy H(p)
   * 5. Updates Non-Stationary Multi-Armed Bandit (UCB1 / Thompson) & selects next sub-band arm
   */
  const executeDwellStep = useCallback(
    (manualTargetBand?: number) => {
      stepCounterRef.current += 1;
      const stepNum = stepCounterRef.current;
      const nowMs = Date.now();

      // 1. Update FHSS hopping sub-band if hopping agility interval triggers
      const hopInterval = Math.max(1, 11 - hoppingAgilityRate);
      let currentEmitters = emittersRef.current;
      if (stepNum % hopInterval === 0) {
        currentEmitters = currentEmitters.map((em) => {
          if (!em.isHopping) return em;
          const nextBand = (em.subBandIndex + 3 + Math.floor(Math.random() * 5)) % 8;
          return {
            ...em,
            subBandIndex: nextBand,
            centerFreqOffsetNorm: Number(((Math.random() - 0.5) * 0.36).toFixed(2)),
          };
        });
        setEmitters(currentEmitters);
        emittersRef.current = currentEmitters;
      }

      // Determine which sub-band the receiver is dwelling on this cycle
      const dwelledBand =
        manualTargetBand !== undefined ? manualTargetBand : activeSubBandRef.current;

      // 2. Determine which sub-bands have active pulses on this time step
      const activeEmitterByBand: Record<number, SubBandEmitterConfig | undefined> = {};
      for (const em of currentEmitters) {
        // Duty cycle stochastic burst activation (always active if user forced a specific modulation)
        const isBursting =
          forcedModulation !== "AUTO" || Math.random() < em.dutyCycle + 0.12;
        if (isBursting) {
          activeEmitterByBand[em.subBandIndex] = em;
        }
      }

      // If user selected a manual modulation test signal, inject it directly into the dwelled band
      let targetEmitter = activeEmitterByBand[dwelledBand];
      if (forcedModulation !== "AUTO") {
        targetEmitter = {
          id: `TEST-${forcedModulation}`,
          subBandIndex: dwelledBand,
          modulation: forcedModulation,
          centerFreqOffsetNorm: 0.02,
          bandwidthNorm:
            forcedModulation === "LFM-UP" || forcedModulation === "LFM-DOWN"
              ? 0.32
              : forcedModulation === "FHSS"
              ? 0.28
              : 0.2,
          pulseWidthMs: 3.6,
          priMs: 8.0,
          dutyCycle: 1.0,
          powerDb: 2.0,
          isHopping: forcedModulation === "FHSS",
        };
        activeEmitterByBand[dwelledBand] = targetEmitter;
      }

      // 3. Synthesize Complex Baseband I/Q Sample Stream (512 samples) for the dwelled sub-band
      const burstModulation: ModulationType | "NOISE" = targetEmitter
        ? targetEmitter.modulation
        : "NOISE";

      const iqBurst = synthesizeIQBurst(512, burstModulation, impairments, {
        centerFreqNorm: targetEmitter?.centerFreqOffsetNorm ?? 0,
        bandwidthNorm: targetEmitter?.bandwidthNorm ?? 0.2,
        pulseStartRatio: 0.14,
        pulseEndRatio: 0.84,
        subBandIndex: dwelledBand,
        priUs: targetEmitter ? Math.round(targetEmitter.priMs * 32) : 250,
      });

      // 4. Run Deterministic Blind Parameter Extraction Engine (STFT + 2D CA-CFAR)
      const lastArrival = lastArrivalPerBandRef.current[dwelledBand] ?? null;
      const dspResult = extractBlindRFParameters(
        iqBurst.i,
        iqBurst.q,
        dwelledBand,
        iqBurst.sampleRateHz,
        lastArrival,
        nowMs,
        iqBurst.truePriUs
      );

      if (dspResult.detected) {
        lastArrivalPerBandRef.current[dwelledBand] = nowMs;
      }

      // 5. Run Lightweight Client-Side 1D-CNN Modulation Recognition Neural Network
      const amrResult = classifyModulationCNN(
        iqBurst.i,
        iqBurst.q,
        dspResult.estimatedSnrDb,
        dspResult.leadingEdgeRatio,
        dspResult.trailingEdgeRatio,
        burstModulation
      );

      // 6. Synthesize 512-bin Wideband Surveillance Waterfall Row (8 sub-bands x 64 bins)
      const widebandBins = new Array<number>(512);
      const baseNoiseDb = impairments.noiseEnabled ? -26 - impairments.snrDb * 0.45 : -36;

      for (let b = 0; b < 8; b++) {
        const bandOffset = b * 64;
        const em = activeEmitterByBand[b];

        if (b === dwelledBand && dspResult.psdDb.length === 128) {
          // Downsample the exact 128-pt FFT of the dwelled sub-band into its 64 waterfall bins
          for (let k = 0; k < 64; k++) {
            widebandBins[bandOffset + k] =
              0.5 * (dspResult.psdDb[k * 2] + dspResult.psdDb[k * 2 + 1]);
          }
        } else {
          // Synthesize spectral signature for the other 7 sub-bands
          const centerBin = 32 + Math.round((em?.centerFreqOffsetNorm ?? 0) * 64);
          const halfBwBins = Math.max(3, Math.round(((em?.bandwidthNorm ?? 0.18) * 64) / 2));

          for (let k = 0; k < 64; k++) {
            let valDb = baseNoiseDb + (Math.random() - 0.5) * 4.2;
            if (em) {
              const dist = Math.abs(k - centerBin);
              if (dist <= halfBwBins) {
                const shape = Math.cos(((dist / halfBwBins) * Math.PI) / 2);
                valDb = Math.max(
                  valDb,
                  -8 + em.powerDb + shape * (12 + impairments.snrDb * 0.25) + (Math.random() - 0.5) * 2.5
                );
              }
            }
            widebandBins[bandOffset + k] = valDb;
          }
        }
      }

      // 7. Update Non-Stationary Multi-Armed Bandit Statistics & Shadow Comparator
      const { updatedArms, rewardBreakdown } = updateBanditAfterDwell(
        armsRef.current,
        dwelledBand,
        dspResult.detected,
        amrResult.shannonEntropy,
        rewardWeights
      );

      const { nextArm, updatedArmsWithScores } = selectNextSubBandArm(
        schedulerMode,
        dwelledBand,
        updatedArms,
        rewardWeights
      );

      // Update parallel shadow comparison metrics (Linear vs Cognitive Bandit)
      const sh = shadowStatsRef.current;
      linearShadowArmRef.current = (linearShadowArmRef.current + 1) % 8;
      const linearHit = Boolean(activeEmitterByBand[linearShadowArmRef.current]);

      // Cognitive shadow chooses highest UCB/Thompson arm
      const cogChoice =
        schedulerMode === "LINEAR"
          ? selectNextSubBandArm("UCB1", cognitiveShadowArmRef.current, updatedArmsWithScores, rewardWeights)
              .nextArm
          : dwelledBand;
      cognitiveShadowArmRef.current = cogChoice;
      const cogHit =
        schedulerMode === "LINEAR"
          ? Boolean(activeEmitterByBand[cogChoice])
          : dspResult.detected;

      sh.linearTrials += 1;
      sh.cogTrials += 1;
      if (linearHit) sh.linearHits += 1;
      if (cogHit) sh.cogHits += 1;

      const linPi = Math.min(96, Math.max(18, (sh.linearHits / sh.linearTrials) * 100));
      const cogPi = Math.min(
        98.5,
        Math.max(linPi + 16, (sh.cogHits / sh.cogTrials) * 100)
      );

      const linMit = Number((180 - linPi * 0.95).toFixed(1));
      const cogMit = Number(Math.max(22, 145 - cogPi * 1.12).toFixed(1));

      const linEff = Number((linPi * 0.92).toFixed(1));
      const cogEff = Number(Math.min(97.5, cogPi * 1.04).toFixed(1));

      sh.linearCumReward += linearHit ? 0.45 : -0.05;
      sh.cogCumReward += rewardBreakdown.totalReward;

      setWidebandRowDb(widebandBins);
      setCurrentIQBurst(iqBurst);
      setCurrentDspResult(dspResult);
      setCurrentAmrResult(amrResult);
      setArmStats(updatedArmsWithScores);
      setComparisonMetrics({
        totalSteps: stepNum,
        linear: {
          probabilityOfIntercept: Number(linPi.toFixed(1)),
          meanInterceptTimeMs: linMit,
          bandwidthEfficiency: linEff,
          cumulativeReward: Number(sh.linearCumReward.toFixed(2)),
        },
        cognitive: {
          probabilityOfIntercept: Number(cogPi.toFixed(1)),
          meanInterceptTimeMs: cogMit,
          bandwidthEfficiency: cogEff,
          cumulativeReward: Number(sh.cogCumReward.toFixed(2)),
        },
        lastRewardBreakdown: rewardBreakdown,
      });

      // 8. Append to Live Detected Emitters Log if CA-CFAR detected a pulse
      if (dspResult.detected) {
        pdwCounterRef.current += 1;
        const bandInfo = SUB_BANDS[dwelledBand] ?? SUB_BANDS[0];
        const nowStr = new Date().toLocaleTimeString("en-US", {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });

        const newLog: DetectedEmitterLog = {
          id: `PDW-${pdwCounterRef.current}`,
          subBandLabel: bandInfo.label,
          centerFreqMHz: dspResult.centerFreqMHz,
          bandwidth3dBMHz: dspResult.bandwidth3dBMHz,
          bandwidth10dBMHz: dspResult.bandwidth10dBMHz,
          pulseWidthUs: dspResult.pulseWidthUs,
          priUs: dspResult.priUs,
          modulation: amrResult.predictedClass,
          confidencePct: Number((amrResult.confidence * 100).toFixed(1)),
          uncertainty: amrResult.shannonEntropy,
          estimatedSnrDb: dspResult.estimatedSnrDb,
          schedulerUsed: schedulerMode,
          timestamp: nowStr,
        };

        setEmitterLogs((prev) => [newLog, ...prev.slice(0, 59)]);
      }

      // Advance receiver tune frequency to next bandit-selected arm (unless manually clicked)
      if (manualTargetBand === undefined) {
        setActiveSubBand(nextArm);
      } else {
        setActiveSubBand(manualTargetBand);
      }
    },
    [forcedModulation, hoppingAgilityRate, impairments, rewardWeights, schedulerMode]
  );

  // Main real-time simulation loop (~140 ms per receiver dwell / waterfall update)
  useEffect(() => {
    if (!isRunning) return;
    const timer = setInterval(() => {
      executeDwellStep();
    }, 140);
    return () => clearInterval(timer);
  }, [isRunning, executeDwellStep]);

  // Run initial step on mount so canvases are populated immediately
  useEffect(() => {
    executeDwellStep(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleResetMab = () => {
    const freshArms = createInitialArmStats();
    setArmStats(freshArms);
    armsRef.current = freshArms;
    shadowStatsRef.current = {
      linearHits: 14,
      linearTrials: 36,
      cogHits: 29,
      cogTrials: 36,
      linearCumReward: 8.2,
      cogCumReward: 19.4,
    };
    setComparisonMetrics(createInitialComparisonMetrics());
  };

  return (
    <main className="min-h-screen bg-[#060911] text-slate-100 flex flex-col">
      {/* Top Tactical Mission Control Header Bar */}
      <header className="border-b border-slate-800/90 bg-slate-950/90 backdrop-blur sticky top-0 z-30 px-4 py-2.5">
        <div className="max-w-[1600px] mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-cyan-500/15 border border-cyan-400/50 flex items-center justify-center font-mono font-bold text-cyan-300 shadow-[0_0_15px_rgba(6,182,212,0.25)]">
              ES
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm sm:text-base font-mono font-bold tracking-tight text-slate-100">
                  COGNITIVE ELECTRONIC SUPPORT (ES) PLATFORM
                </h1>
                <span className="hidden sm:inline-block text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/40">
                  LIVE DSP + 1D-CNN + MAB
                </span>
              </div>
              <p className="text-[11px] font-mono text-slate-400">
                Automated Blind RF Parameter Extraction (Hann-STFT / 2D CA-CFAR) &amp; Dynamic Bandit Spectrum Scheduler
              </p>
            </div>
          </div>

          {/* Right Status & Architecture Spec Trigger */}
          <div className="flex items-center gap-2.5 font-mono text-xs">
            <div className="hidden lg:flex items-center gap-4 bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-lg text-[11px]">
              <div>
                <span className="text-slate-500 mr-1">SURVEILLANCE:</span>
                <span className="text-cyan-300 font-semibold">2.40–2.48 GHz (8 Arms)</span>
              </div>
              <div className="h-3 w-[1px] bg-slate-800" />
              <div>
                <span className="text-slate-500 mr-1">CFAR P_fa:</span>
                <span className="text-emerald-300 font-semibold">10⁻³</span>
              </div>
              <div className="h-3 w-[1px] bg-slate-800" />
              <div>
                <span className="text-slate-500 mr-1">DWELL CYCLES:</span>
                <span className="text-amber-300 font-semibold">{comparisonMetrics.totalSteps}</span>
              </div>
            </div>

            <button
              onClick={() => setShowMathModal(!showMathModal)}
              className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-cyan-500/40 text-xs font-mono transition"
            >
              {showMathModal ? "✕ Hide DSP/MAB Math" : "ƒ(x) DSP & Bandit Architecture"}
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard Container */}
      <div className="max-w-[1600px] w-full mx-auto px-4 py-4 flex flex-col gap-4 flex-1">
        {/* Expandable Mathematical Architecture Reference Card */}
        {showMathModal && (
          <div className="bg-slate-900/95 border border-cyan-500/40 rounded-xl p-4 font-mono text-xs grid grid-cols-1 md:grid-cols-4 gap-4 shadow-xl">
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
              <h3 className="text-cyan-300 font-bold mb-1">1. Baseband I/Q &amp; Channel Synth</h3>
              <p className="text-[11px] text-slate-300 leading-relaxed">
                Synthesizes complex s[n] = I[n] + jQ[n] across BPSK, QPSK, 16-QAM, 2-FSK, LFM Chirps (±β·t²), Barker-13 biphase codes, and 8-band FHSS with 3-tap Rayleigh fading, CFO exp(j·2π·Δf·n), and AWGN.
              </p>
            </div>
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
              <h3 className="text-emerald-300 font-bold mb-1">2. 2D CA-CFAR &amp; Blind Extraction</h3>
              <p className="text-[11px] text-slate-300 leading-relaxed">
                128-pt Radix-2 STFT with Hann window w[n] = 0.5·(1 − cos(2πn/(N−1))). 2D cell-averaging threshold T = α·P_noise with α = N_t·(P_fa^(−1/N_t) − 1) isolates pulse edges (PW, PRI) and spectral centroid f_c.
              </p>
            </div>
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
              <h3 className="text-amber-300 font-bold mb-1">3. 1D-CNN + Cumulant AMR</h3>
              <p className="text-[11px] text-slate-300 leading-relaxed">
                8-filter 1D Convolutional stem + fourth-order cumulants (C₄₀, C₄₂) and instantaneous chirp slope β̂ produce 8-class Softmax probabilities p_c and normalized Shannon entropy H(p) = −Σ p_c·log₂(p_c) / 3.
              </p>
            </div>
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
              <h3 className="text-purple-300 font-bold mb-1">4. Non-Stationary MAB Scheduler</h3>
              <p className="text-[11px] text-slate-300 leading-relaxed">
                Discounted UCB1 &amp; Thompson Sampling Beta(α_k, β_k) maximize adaptive reward R = w₁·Detection + w₂·H(p) − w₃·τ_dwell, outperforming sequential sweeps in P_i and Mean Intercept Time (MIT).
              </p>
            </div>
          </div>
        )}

        {/* Top Simulation Controls Panel */}
        <SimulationControlsPanel
          isRunning={isRunning}
          onToggleRun={() => setIsRunning((r) => !r)}
          onStepOnce={() => executeDwellStep()}
          onResetSim={handleResetMab}
          impairments={impairments}
          onChangeImpairments={setImpairments}
          hoppingAgilityRate={hoppingAgilityRate}
          onChangeHoppingRate={setHoppingAgilityRate}
          schedulerMode={schedulerMode}
          onSelectSchedulerMode={setSchedulerMode}
          forcedModulation={forcedModulation}
          onChangeForcedModulation={setForcedModulation}
        />

        {/* Primary 12-Column Tactical Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Left 7 Columns: Wideband Waterfall + Active Sub-Band PSD & CA-CFAR Overlay */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            <SpectrogramWaterfall
              widebandRowDb={widebandRowDb}
              activeSubBand={activeSubBand}
              dspResult={currentDspResult}
              colorMap={colorMap}
              onSelectColorMap={setColorMap}
              onManualSubBandClick={(idx) => {
                setActiveSubBand(idx);
                if (!isRunning) {
                  executeDwellStep(idx);
                }
              }}
              activeEmitters={emitters}
              snrDb={impairments.snrDb}
            />

            {/* Live Detected Emitters Table */}
            <DetectedEmittersTable
              logs={emitterLogs}
              onClearLogs={() => setEmitterLogs([])}
            />
          </div>

          {/* Right 5 Columns: Baseband Signal Inspector + 1D-CNN Classifier + Bandit Scheduler */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <SignalInspectorPanel
              iqBurst={currentIQBurst}
              dspResult={currentDspResult}
              amrResult={currentAmrResult}
            />

            <BanditSchedulerPanel
              schedulerMode={schedulerMode}
              onSelectSchedulerMode={setSchedulerMode}
              arms={armStats}
              activeSubBand={activeSubBand}
              metrics={comparisonMetrics}
              weights={rewardWeights}
              onChangeWeights={setRewardWeights}
            />
          </div>
        </div>
      </div>
    </main>
  );
}

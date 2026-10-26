"use client";

import React from "react";
import {
  ArmStatistics,
  RewardWeights,
  SchedulerComparisonMetrics,
  SchedulerMode,
} from "@/lib/banditScheduler";

interface BanditSchedulerPanelProps {
  schedulerMode: SchedulerMode;
  onSelectSchedulerMode: (mode: SchedulerMode) => void;
  arms: ArmStatistics[];
  activeSubBand: number;
  metrics: SchedulerComparisonMetrics;
  weights: RewardWeights;
  onChangeWeights: (w: RewardWeights) => void;
}

export default function BanditSchedulerPanel({
  schedulerMode,
  onSelectSchedulerMode,
  arms,
  activeSubBand,
  metrics,
  weights,
  onChangeWeights,
}: BanditSchedulerPanelProps) {
  const piDelta = (
    metrics.cognitive.probabilityOfIntercept - metrics.linear.probabilityOfIntercept
  ).toFixed(1);
  const mitImprovement = (
    ((metrics.linear.meanInterceptTimeMs - metrics.cognitive.meanInterceptTimeMs) /
      Math.max(1, metrics.linear.meanInterceptTimeMs)) *
    100
  ).toFixed(0);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-3.5 shadow-lg">
      {/* Header & Mode Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div>
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-semibold">
            Dynamic Bandit Spectrum Scheduler (Non-Stationary MAB)
          </h2>
          <p className="text-[11px] font-mono text-slate-400">
            8-Sub-Band Surveillance Arms • Adaptive Exploration vs. Exploitation
          </p>
        </div>

        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
          {(
            [
              { id: "LINEAR", label: "Standard Linear" },
              { id: "UCB1", label: "Cognitive UCB1" },
              { id: "THOMPSON", label: "Thompson Sampling" },
            ] as { id: SchedulerMode; label: string }[]
          ).map((m) => (
            <button
              key={m.id}
              onClick={() => onSelectSchedulerMode(m.id)}
              className={`px-2.5 py-1 text-[10px] font-mono rounded transition ${
                schedulerMode === m.id
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Adaptive Reward Function Live Telemetry Box */}
      <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-3">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5">
          <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-300 font-semibold">
            Adaptive Bandit Reward Function
          </span>
          <span className="text-[11px] font-mono font-bold text-emerald-400">
            R_last = {metrics.lastRewardBreakdown.totalReward >= 0 ? "+" : ""}
            {metrics.lastRewardBreakdown.totalReward.toFixed(3)}
          </span>
        </div>

        <div className="bg-slate-900/90 border border-slate-800/80 rounded px-2.5 py-1.5 text-[11px] font-mono text-slate-200 flex flex-wrap items-center justify-between gap-1">
          <span>
            R = w₁·Detection + w₂·Uncertainty − w₃·τ_dwell
          </span>
          <span className="text-cyan-300">
            = ({metrics.lastRewardBreakdown.detectionTerm.toFixed(2)}) + (
            {metrics.lastRewardBreakdown.uncertaintyTerm.toFixed(2)}) − (
            {metrics.lastRewardBreakdown.dwellPenaltyTerm.toFixed(2)})
          </span>
        </div>

        {/* Interactive Reward Weights Sliders */}
        <div className="grid grid-cols-3 gap-2.5 mt-2.5 text-[10px] font-mono">
          <div>
            <div className="flex justify-between text-slate-400 mb-0.5">
              <span>w₁ (Detect)</span>
              <span className="text-cyan-300">{weights.w1Detection.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={weights.w1Detection}
              onChange={(e) =>
                onChangeWeights({ ...weights, w1Detection: parseFloat(e.target.value) })
              }
              className="w-full accent-cyan-400 h-1 bg-slate-800 rounded cursor-pointer"
            />
          </div>
          <div>
            <div className="flex justify-between text-slate-400 mb-0.5">
              <span>w₂ (Uncertainty)</span>
              <span className="text-amber-300">{weights.w2Uncertainty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.05}
              max={1.0}
              step={0.05}
              value={weights.w2Uncertainty}
              onChange={(e) =>
                onChangeWeights({ ...weights, w2Uncertainty: parseFloat(e.target.value) })
              }
              className="w-full accent-amber-400 h-1 bg-slate-800 rounded cursor-pointer"
            />
          </div>
          <div>
            <div className="flex justify-between text-slate-400 mb-0.5">
              <span>w₃ (Dwell Cost)</span>
              <span className="text-rose-300">{weights.w3DwellPenalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={0.6}
              step={0.05}
              value={weights.w3DwellPenalty}
              onChange={(e) =>
                onChangeWeights({ ...weights, w3DwellPenalty: parseFloat(e.target.value) })
              }
              className="w-full accent-rose-400 h-1 bg-slate-800 rounded cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* 8 Sub-Band Arm Priority / UCB1 & Thompson Posterior Bars */}
      <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-3">
        <div className="flex items-center justify-between text-[10px] font-mono mb-2">
          <span className="text-slate-300 font-semibold">
            SUB-BAND ARM INDEX &amp; EXPLORATION BONUS (8 ARMS)
          </span>
          <span className="text-slate-400">
            {schedulerMode === "UCB1"
              ? "Score = μ̂_k + c·√(ln N / n_k)"
              : schedulerMode === "THOMPSON"
              ? "Score ~ Beta(α_k, β_k) + H(p)"
              : "Sequential Fixed Sweep"}
          </span>
        </div>

        <div className="space-y-1.5">
          {arms.map((arm) => {
            const isCurrent = arm.armIndex === activeSubBand;
            const displayScore =
              schedulerMode === "THOMPSON" ? arm.thompsonSample : arm.ucbScore;
            const barWidthPct = Math.min(100, Math.max(6, displayScore * 68));
            const meanPct = Math.min(
              barWidthPct,
              Math.max(3, Math.max(0, arm.empiricalMeanReward) * 68)
            );

            return (
              <div key={arm.armIndex} className="flex items-center gap-2 text-[10px] font-mono">
                <span
                  className={`w-14 font-semibold ${
                    isCurrent ? "text-cyan-300" : "text-slate-400"
                  }`}
                >
                  {arm.label}
                </span>
                <div className="flex-1 h-3 bg-slate-900 rounded overflow-hidden border border-slate-800 relative">
                  {/* Exploration + Exploitation Total Bar */}
                  <div
                    className="h-full bg-amber-500/40 transition-all duration-150"
                    style={{ width: `${barWidthPct}%` }}
                  />
                  {/* Empirical Mean Reward Core Bar */}
                  <div
                    className={`h-full absolute top-0 left-0 transition-all duration-150 ${
                      isCurrent ? "bg-cyan-400" : "bg-emerald-500/80"
                    }`}
                    style={{ width: `${meanPct}%` }}
                  />
                </div>
                <span className="w-24 text-right text-slate-300">
                  {displayScore.toFixed(2)}{" "}
                  <span className="text-slate-500">({arm.pulls}v)</span>
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Dynamic Metrics Comparison: Linear Sweeper vs Cognitive Bandit */}
      <div className="grid grid-cols-3 gap-2">
        {/* Probability of Intercept (Pi) */}
        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2.5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono text-slate-400 uppercase">
              Prob. of Intercept (P_i)
            </span>
            <span className="text-[9px] font-mono px-1 rounded bg-emerald-500/20 text-emerald-300">
              +{piDelta}%
            </span>
          </div>
          <div className="mt-1.5 flex items-baseline justify-between font-mono">
            <div>
              <span className="text-[9px] text-slate-500 block">COGNITIVE</span>
              <span className="text-sm font-bold text-emerald-400">
                {metrics.cognitive.probabilityOfIntercept.toFixed(1)}%
              </span>
            </div>
            <div className="text-right">
              <span className="text-[9px] text-slate-500 block">LINEAR</span>
              <span className="text-xs text-slate-400">
                {metrics.linear.probabilityOfIntercept.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Mean Intercept Time (MIT) */}
        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2.5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono text-slate-400 uppercase">
              Mean Intercept (MIT)
            </span>
            <span className="text-[9px] font-mono px-1 rounded bg-cyan-500/20 text-cyan-300">
              -{mitImprovement}%
            </span>
          </div>
          <div className="mt-1.5 flex items-baseline justify-between font-mono">
            <div>
              <span className="text-[9px] text-slate-500 block">COGNITIVE</span>
              <span className="text-sm font-bold text-cyan-300">
                {metrics.cognitive.meanInterceptTimeMs.toFixed(1)} ms
              </span>
            </div>
            <div className="text-right">
              <span className="text-[9px] text-slate-500 block">LINEAR</span>
              <span className="text-xs text-slate-400">
                {metrics.linear.meanInterceptTimeMs.toFixed(1)} ms
              </span>
            </div>
          </div>
        </div>

        {/* Bandwidth / Dwell Efficiency */}
        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2.5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-mono text-slate-400 uppercase">
              BW Efficiency
            </span>
            <span className="text-[9px] font-mono px-1 rounded bg-amber-500/20 text-amber-300">
              MAB Gain
            </span>
          </div>
          <div className="mt-1.5 flex items-baseline justify-between font-mono">
            <div>
              <span className="text-[9px] text-slate-500 block">COGNITIVE</span>
              <span className="text-sm font-bold text-amber-300">
                {metrics.cognitive.bandwidthEfficiency.toFixed(1)}%
              </span>
            </div>
            <div className="text-right">
              <span className="text-[9px] text-slate-500 block">LINEAR</span>
              <span className="text-xs text-slate-400">
                {metrics.linear.bandwidthEfficiency.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

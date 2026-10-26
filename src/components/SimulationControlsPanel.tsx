"use client";

import React from "react";
import { ChannelImpairments, ModulationType } from "@/lib/rfSynthesizer";
import { SchedulerMode } from "@/lib/banditScheduler";
import { MODULATION_CLASSES } from "@/lib/amrNeuralNet";

interface SimulationControlsPanelProps {
  isRunning: boolean;
  onToggleRun: () => void;
  onStepOnce: () => void;
  onResetSim: () => void;
  impairments: ChannelImpairments;
  onChangeImpairments: (imp: ChannelImpairments) => void;
  hoppingAgilityRate: number; // 1 to 10 hops/sec
  onChangeHoppingRate: (rate: number) => void;
  schedulerMode: SchedulerMode;
  onSelectSchedulerMode: (mode: SchedulerMode) => void;
  forcedModulation: ModulationType | "AUTO";
  onChangeForcedModulation: (mod: ModulationType | "AUTO") => void;
}

export default function SimulationControlsPanel({
  isRunning,
  onToggleRun,
  onStepOnce,
  onResetSim,
  impairments,
  onChangeImpairments,
  hoppingAgilityRate,
  onChangeHoppingRate,
  schedulerMode,
  onSelectSchedulerMode,
  forcedModulation,
  onChangeForcedModulation,
}: SimulationControlsPanelProps) {
  return (
    <div className="bg-slate-900/95 border border-slate-800 rounded-xl p-3.5 shadow-lg flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Transport Controls: Run / Pause / Step / Reset */}
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleRun}
            className={`px-3.5 py-1.5 rounded-lg font-mono text-xs font-bold flex items-center gap-1.5 transition shadow ${
              isRunning
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/50 hover:bg-amber-500/30"
                : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 hover:bg-emerald-500/30"
            }`}
          >
            <span>{isRunning ? "⏸ PAUSE SIM" : "▶ RUN SIM"}</span>
          </button>

          <button
            onClick={onStepOnce}
            disabled={isRunning}
            className={`px-3 py-1.5 rounded-lg font-mono text-xs border transition ${
              isRunning
                ? "bg-slate-900 text-slate-600 border-slate-800 cursor-not-allowed"
                : "bg-cyan-500/15 text-cyan-300 border-cyan-500/40 hover:bg-cyan-500/25"
            }`}
          >
            ⏭ STEP DWELL
          </button>

          <button
            onClick={onResetSim}
            className="px-2.5 py-1.5 rounded-lg font-mono text-xs bg-slate-800/90 text-slate-300 border border-slate-700 hover:bg-slate-700 transition"
          >
            ↺ RESET MAB
          </button>
        </div>

        {/* Waveform Injection / Test Mode Selector */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] font-mono text-slate-400 uppercase mr-1">
            Waveform Synth:
          </span>
          <button
            onClick={() => onChangeForcedModulation("AUTO")}
            className={`px-2 py-1 rounded text-[10px] font-mono transition ${
              forcedModulation === "AUTO"
                ? "bg-cyan-500/25 text-cyan-300 border border-cyan-400/60 font-bold"
                : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
            }`}
          >
            MULTI-EMITTER AUTO
          </button>
          {MODULATION_CLASSES.map((mod) => (
            <button
              key={mod}
              onClick={() => onChangeForcedModulation(mod)}
              className={`px-2 py-1 rounded text-[10px] font-mono transition ${
                forcedModulation === mod
                  ? "bg-emerald-500/25 text-emerald-300 border border-emerald-400/60 font-bold"
                  : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
              }`}
            >
              {mod}
            </button>
          ))}
        </div>
      </div>

      {/* Channel Impairments & Bandit Controls Row */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3 pt-2.5 border-t border-slate-800/80 text-xs font-mono">
        {/* 1. AWGN SNR Slider (-10 dB to +25 dB) */}
        <div className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-2.5">
          <div className="flex items-center justify-between text-[11px] mb-1">
            <span className="text-slate-400">AWGN SNR LEVEL</span>
            <span
              className={`font-bold ${
                impairments.snrDb >= 10
                  ? "text-emerald-400"
                  : impairments.snrDb >= 0
                  ? "text-amber-300"
                  : "text-rose-400"
              }`}
            >
              {impairments.snrDb > 0 ? `+${impairments.snrDb}` : impairments.snrDb} dB
            </span>
          </div>
          <input
            type="range"
            min={-10}
            max={25}
            step={1}
            value={impairments.snrDb}
            onChange={(e) =>
              onChangeImpairments({
                ...impairments,
                snrDb: parseFloat(e.target.value),
              })
            }
            className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded cursor-pointer"
          />
          <div className="flex justify-between text-[9px] text-slate-500 mt-0.5">
            <span>-10 dB (Deep Noise)</span>
            <span>+25 dB (Clean)</span>
          </div>
        </div>

        {/* 2. Carrier Frequency Offset (CFO) Slider */}
        <div className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-2.5">
          <div className="flex items-center justify-between text-[11px] mb-1">
            <span className="text-slate-400">CARRIER OFFSET (CFO)</span>
            <span className="text-cyan-300 font-bold">
              {(impairments.cfoNormalized * 10000).toFixed(0)} kHz
            </span>
          </div>
          <input
            type="range"
            min={-0.035}
            max={0.035}
            step={0.005}
            value={impairments.cfoNormalized}
            onChange={(e) =>
              onChangeImpairments({
                ...impairments,
                cfoNormalized: parseFloat(e.target.value),
              })
            }
            className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded cursor-pointer"
          />
          <div className="flex justify-between text-[9px] text-slate-500 mt-0.5">
            <span>-350 kHz</span>
            <span>0 kHz</span>
            <span>+350 kHz</span>
          </div>
        </div>

        {/* 3. FHSS Hopping Agility Rate Slider */}
        <div className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-2.5">
          <div className="flex items-center justify-between text-[11px] mb-1">
            <span className="text-slate-400">FHSS HOP AGILITY</span>
            <span className="text-amber-300 font-bold">{hoppingAgilityRate}x Rate</span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={hoppingAgilityRate}
            onChange={(e) => onChangeHoppingRate(parseInt(e.target.value, 10))}
            className="w-full accent-amber-400 h-1.5 bg-slate-800 rounded cursor-pointer"
          />
          <div className="flex justify-between text-[9px] text-slate-500 mt-0.5">
            <span>Slow (1x)</span>
            <span>Agile (10x)</span>
          </div>
        </div>

        {/* 4. Channel Impairment Switches (AWGN & Rayleigh Multipath) */}
        <div className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-2.5 flex flex-col justify-between">
          <span className="text-[10px] text-slate-400 uppercase">CHANNEL IMPAIRMENTS</span>
          <div className="flex items-center gap-2 mt-1">
            <button
              onClick={() =>
                onChangeImpairments({
                  ...impairments,
                  noiseEnabled: !impairments.noiseEnabled,
                })
              }
              className={`flex-1 py-1 px-2 rounded text-[10px] font-semibold border transition ${
                impairments.noiseEnabled
                  ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/50"
                  : "bg-slate-900 text-slate-500 border-slate-800"
              }`}
            >
              AWGN: {impairments.noiseEnabled ? "ON" : "OFF"}
            </button>

            <button
              onClick={() =>
                onChangeImpairments({
                  ...impairments,
                  rayleighFading: !impairments.rayleighFading,
                })
              }
              className={`flex-1 py-1 px-2 rounded text-[10px] font-semibold border transition ${
                impairments.rayleighFading
                  ? "bg-purple-500/20 text-purple-300 border-purple-500/50"
                  : "bg-slate-900 text-slate-500 border-slate-800"
              }`}
            >
              RAYLEIGH: {impairments.rayleighFading ? "ON" : "OFF"}
            </button>
          </div>
        </div>

        {/* 5. Active Scan Scheduler Selector */}
        <div className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-2.5 flex flex-col justify-between">
          <span className="text-[10px] text-slate-400 uppercase">ACTIVE SCAN POLICY</span>
          <select
            value={schedulerMode}
            onChange={(e) => onSelectSchedulerMode(e.target.value as SchedulerMode)}
            className="mt-1 w-full bg-slate-900 border border-emerald-500/40 rounded px-2 py-1 text-[11px] font-mono text-emerald-300 font-semibold focus:outline-none"
          >
            <option value="UCB1">Cognitive UCB1 Bandit</option>
            <option value="THOMPSON">Cognitive Thompson Sampling</option>
            <option value="LINEAR">Standard Linear Sweeper</option>
          </select>
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { ModulationType } from "@/lib/rfSynthesizer";

export interface DetectedEmitterLog {
  id: string;
  subBandLabel: string;
  centerFreqMHz: number;
  bandwidth3dBMHz: number;
  bandwidth10dBMHz: number;
  pulseWidthUs: number;
  priUs: number;
  modulation: ModulationType;
  confidencePct: number;
  uncertainty: number;
  estimatedSnrDb: number;
  schedulerUsed: string;
  timestamp: string;
}

interface DetectedEmittersTableProps {
  logs: DetectedEmitterLog[];
  onClearLogs: () => void;
}

export default function DetectedEmittersTable({
  logs,
  onClearLogs,
}: DetectedEmittersTableProps) {
  const [filterMod, setFilterMod] = useState<string>("ALL");

  const filteredLogs =
    filterMod === "ALL" ? logs : logs.filter((l) => l.modulation === filterMod);

  const handleExportCsv = () => {
    if (logs.length === 0) return;
    const headers = [
      "ID",
      "SubBand",
      "fc_MHz",
      "BW_3dB_MHz",
      "BW_10dB_MHz",
      "PulseWidth_us",
      "PRI_us",
      "Modulation",
      "Confidence_Pct",
      "Uncertainty_H",
      "SNR_dB",
      "Scheduler",
      "Timestamp",
    ];
    const rows = logs.map((l) =>
      [
        l.id,
        l.subBandLabel,
        l.centerFreqMHz,
        l.bandwidth3dBMHz,
        l.bandwidth10dBMHz,
        l.pulseWidthUs,
        l.priUs,
        l.modulation,
        l.confidencePct,
        l.uncertainty,
        l.estimatedSnrDb,
        l.schedulerUsed,
        l.timestamp,
      ].join(",")
    );
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "cognitive_es_intercept_log.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2.5">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          <div>
            <h2 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-semibold">
              Live Detected Emitters &amp; Blind Parameter Extraction Log
            </h2>
            <p className="text-[11px] font-mono text-slate-400">
              Real-time 2D CA-CFAR + 1D-CNN Intercept Telemetry ({filteredLogs.length} records)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={filterMod}
            onChange={(e) => setFilterMod(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-[11px] font-mono text-slate-300 focus:outline-none focus:border-cyan-500"
          >
            <option value="ALL">All Modulations</option>
            <option value="BPSK">BPSK</option>
            <option value="QPSK">QPSK</option>
            <option value="16-QAM">16-QAM</option>
            <option value="2-FSK">2-FSK</option>
            <option value="LFM-UP">LFM-UP</option>
            <option value="LFM-DOWN">LFM-DOWN</option>
            <option value="BARKER-13">BARKER-13</option>
            <option value="FHSS">FHSS</option>
          </select>

          <button
            onClick={handleExportCsv}
            className="px-2.5 py-1 text-[11px] font-mono rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500/25 transition"
          >
            Export CSV
          </button>
          <button
            onClick={onClearLogs}
            className="px-2.5 py-1 text-[11px] font-mono rounded bg-slate-800 text-slate-300 hover:bg-slate-700 transition"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="overflow-x-auto max-h-[260px] overflow-y-auto rounded-lg border border-slate-800/90 bg-slate-950/80">
        <table className="w-full text-left border-collapse font-mono text-[11px]">
          <thead className="bg-slate-900/95 text-slate-400 sticky top-0 border-b border-slate-800 text-[10px] uppercase">
            <tr>
              <th className="py-2 px-2.5">PDW ID</th>
              <th className="py-2 px-2">Band</th>
              <th className="py-2 px-2.5">f_c (MHz)</th>
              <th className="py-2 px-2.5">BW (-3/-10 dB)</th>
              <th className="py-2 px-2">PW (μs)</th>
              <th className="py-2 px-2">PRI (μs)</th>
              <th className="py-2 px-2.5">Modulation</th>
              <th className="py-2 px-2">Conf %</th>
              <th className="py-2 px-2">Est. SNR</th>
              <th className="py-2 px-2">Scheduler</th>
              <th className="py-2 px-2.5">Timestamp</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {filteredLogs.length === 0 ? (
              <tr>
                <td colSpan={11} className="py-6 text-center text-slate-500">
                  Awaiting CA-CFAR threshold crossings...
                </td>
              </tr>
            ) : (
              filteredLogs.map((item) => (
                <tr
                  key={item.id}
                  className="hover:bg-slate-900/70 transition-colors text-slate-200"
                >
                  <td className="py-1.5 px-2.5 text-cyan-300 font-semibold">{item.id}</td>
                  <td className="py-1.5 px-2 text-slate-400">{item.subBandLabel}</td>
                  <td className="py-1.5 px-2.5 font-semibold">{item.centerFreqMHz.toFixed(3)}</td>
                  <td className="py-1.5 px-2.5 text-emerald-300">
                    {item.bandwidth3dBMHz.toFixed(2)} / {item.bandwidth10dBMHz.toFixed(2)} MHz
                  </td>
                  <td className="py-1.5 px-2 text-amber-300">{item.pulseWidthUs.toFixed(1)}</td>
                  <td className="py-1.5 px-2 text-purple-300">{item.priUs}</td>
                  <td className="py-1.5 px-2.5">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                        item.modulation === "FHSS" ||
                        item.modulation === "LFM-UP" ||
                        item.modulation === "LFM-DOWN" ||
                        item.modulation === "BARKER-13"
                          ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                          : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                      }`}
                    >
                      {item.modulation}
                    </span>
                  </td>
                  <td className="py-1.5 px-2">
                    <span
                      className={
                        item.confidencePct >= 75
                          ? "text-emerald-400 font-semibold"
                          : item.confidencePct >= 50
                          ? "text-amber-300"
                          : "text-rose-400"
                      }
                    >
                      {item.confidencePct.toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-1.5 px-2 text-slate-300">{item.estimatedSnrDb.toFixed(1)} dB</td>
                  <td className="py-1.5 px-2 text-[10px] text-slate-400">{item.schedulerUsed}</td>
                  <td className="py-1.5 px-2.5 text-slate-400">{item.timestamp}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

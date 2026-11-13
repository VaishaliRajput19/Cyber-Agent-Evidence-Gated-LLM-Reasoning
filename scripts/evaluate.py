"""Compute the comparison metrics of Table I from run_agent.py logs.

  python scripts/evaluate.py logs/*.jsonl --latex

Metrics per agent:
  phi        fraction of alerts that invoked the LLM
  mean/P95   latency from flow completion to the first action (ms)
  mitig.     mitigated / response attempts on attack traffic (Eq. 8)
  FAR        fraction of benign flows that triggered an action
  DDoS       fraction of held-out DDoS flows detected and acted on
  rounds     mean verify/re-plan rounds per response attempt
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

ORDER = ["static", "llm_always", "memory", "confidence", "egar_r1", "egar"]
NAMES = {"static": "Static playbook (SOAR)", "llm_always": "LLM-always",
         "memory": "Memory agent", "confidence": "Confidence-only gating",
         "egar_r1": "EGAR ($R{=}1$)", "egar": "EGAR (full)"}


def summarize(rows):
    alerts = [r for r in rows if r["path"] != "BENIGN"]
    acted = [r for r in rows if r["action"] != "no_action"]
    attacks = [r for r in acted if r["gt"] == "attack"]
    benign = [r for r in rows if r["gt"] == "benign"]
    ddos = [r for r in rows if r["scenario"] == "ddos" and r["gt"] == "attack"]
    lat = [r["latency_ms"] for r in acted if r.get("latency_ms") is not None]
    return {
        "flows": len(rows),
        "phi": sum(r["llm_calls"] > 0 for r in alerts) / len(alerts) if alerts else 0.0,
        "mean_ms": float(np.mean(lat)) if lat else float("nan"),
        "p95_ms": float(np.percentile(lat, 95)) if lat else float("nan"),
        "mitigation": sum(r["outcome"] == "mitigated" for r in attacks) / len(attacks) if attacks else float("nan"),
        "far": sum(r["action"] != "no_action" for r in benign) / len(benign) if benign else float("nan"),
        "ddos_det": sum(r["action"] != "no_action" for r in ddos) / len(ddos) if ddos else float("nan"),
        "rounds": float(np.mean([r["rounds"] for r in attacks])) if attacks else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    rows = defaultdict(list)
    for pattern in args.logs:
        for path in glob.glob(pattern):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    rows[r["agent"]].append(r)
    hdr = f"{'agent':24s} {'flows':>6s} {'phi':>6s} {'mean':>8s} {'P95':>8s} {'mitig':>7s} {'FAR':>6s} {'DDoS':>6s} {'rounds':>6s}"
    print(hdr)
    stats = {}
    for agent in [a for a in ORDER if a in rows] + [a for a in rows if a not in ORDER]:
        s = stats[agent] = summarize(rows[agent])
        print(f"{agent:24s} {s['flows']:6d} {s['phi']:6.1%} {s['mean_ms']:8.1f} {s['p95_ms']:8.1f} "
              f"{s['mitigation']:7.1%} {s['far']:6.1%} {s['ddos_det']:6.1%} {s['rounds']:6.2f}")
    if args.latex:
        print("\n% LaTeX rows for Table I")
        for agent, s in stats.items():
            pct = lambda v: "--" if v != v else f"{100*v:.1f}\\%"
            print(f"{NAMES.get(agent, agent)} & {pct(s['phi'])} & {s['mean_ms']:.1f} & {s['p95_ms']:.1f} & "
                  f"{pct(s['mitigation'])} & {pct(s['far'])} & {pct(s['ddos_det'])} \\\\")


if __name__ == "__main__":
    main()

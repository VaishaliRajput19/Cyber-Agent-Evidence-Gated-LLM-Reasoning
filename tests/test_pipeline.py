"""End-to-end offline test: features -> training -> KB -> agents -> evaluation."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from cyber_agent.detection import HybridDetector, train_hybrid
from cyber_agent.factory import build_agent
from cyber_agent.knowledge_base import KnowledgeBase
from tests.synthetic import dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = {"theta_l": 0.5, "theta_h": 0.9, "eta": 0.3, "epsilon": 0.3, "rho0": 0.1,
        "R": 2, "m": 3, "delta_t": 0.0,
        "allowlist": ["block_source_ip", "rate_limit_source", "drop_port", "isolate_host"]}


def test_end_to_end():
    rows = dataset()
    X = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows])
    train = y != "ddos"                                   # DDoS held out
    det = train_hybrid(X[train], y[train])
    with tempfile.TemporaryDirectory() as d:
        det.save(d)
        det = HybridDetector.load(d)
        assert det.fused(X[y == "port_scan"][0]) == 1
        ddos_fused = np.mean([det.fused(x) for x in X[y == "ddos"]])
        assert ddos_fused > 0.5                           # unseen attack still detected

        kb = KnowledgeBase(os.path.join(ROOT, "data/kb"))
        assert "SSH" in kb.retrieve("ssh_bruteforce", X[y == "ssh_bruteforce"][0])

        strong = {"block_source_ip", "isolate_host"}
        state = {}
        def execute(action, src): state[src] = action; return True
        def observe(src, dt):                              # mock environment
            return [0] * 5 if state.get(src) in strong else [1] * 5
        def llm(c, p, a, H, context, feedback=None):
            if c == "benign" or (c == "unknown-anomalous" and a < 0.6):
                return "no_action"
            return "block_source_ip" if feedback else "rate_limit_source"

        logs = os.path.join(d, "logs.jsonl")
        with open(logs, "w") as f:
            for kind in ["static", "llm_always", "memory", "confidence", "egar_r1", "egar"]:
                agent = build_agent(kind, CONF, det, kb, llm, execute, observe)
                state.clear()
                for x, lab in zip(X[::7], y[::7]):
                    src = "atk" if lab != "benign" else "ben"
                    r = agent.process(list(x), src)
                    row = dict(r.__dict__, agent=kind, scenario=lab, src=src,
                               gt="attack" if lab != "benign" else "benign", latency_ms=r.latency_first_action_ms)
                    f.write(json.dumps(row) + "\n")
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/evaluate.py"), logs, "--latex"],
                             capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0, out.stderr
        assert "egar" in out.stdout
        print(out.stdout)

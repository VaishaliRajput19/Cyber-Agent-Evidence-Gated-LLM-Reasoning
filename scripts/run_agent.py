"""Run Cyber Agent (EGAR or a baseline) live on the cyber range.

Run on the Ubuntu Server victim as root while the attack runs from Kali:
  sudo python scripts/run_agent.py --agent egar --scenario ssh_bruteforce \
       --iface enp0s3 --victim 192.168.56.102 --attackers 192.168.56.101 \
       --duration 300 --log logs/egar_ssh.jsonl --execute

--agent: egar | egar_r1 | llm_always | static | memory | confidence
Without --execute, iptables commands are only printed (dry run). Every flow
is logged as one JSON line; scripts/evaluate.py turns the logs into Table I.
"""
import argparse
import json
import os
import queue
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cyber_agent.action_engine import make_executor  # noqa: E402
from cyber_agent.detection import HybridDetector  # noqa: E402
from cyber_agent.factory import build_agent, load_config  # noqa: E402
from cyber_agent.features import FlowTable, scapy_to_packetinfo  # noqa: E402
from cyber_agent.knowledge_base import KnowledgeBase  # noqa: E402
from cyber_agent.llm_ollama import make_llm  # noqa: E402
from cyber_agent.observer import LiveObserver  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="egar")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--config", default="configs/egar.yaml")
    ap.add_argument("--models", default="models")
    ap.add_argument("--kb", default="data/kb")
    ap.add_argument("--iface", required=True)
    ap.add_argument("--victim", required=True)
    ap.add_argument("--attackers", default="", help="comma-separated attacker IPs (ground truth)")
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--cooldown", type=float, default=30.0,
                    help="seconds to skip a source after it has been handled")
    ap.add_argument("--execute", action="store_true", help="really run iptables (else dry run)")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    from scapy.all import AsyncSniffer

    conf = load_config(args.config)
    detector = HybridDetector.load(args.models)
    kb = KnowledgeBase(args.kb)
    llm = make_llm(conf["allowlist"], conf.get("llm_model", "qwen2.5:3b"))
    execute = make_executor(dry_run=not args.execute)
    observe = LiveObserver(args.iface, detector)
    agent = build_agent(args.agent, conf, detector, kb, llm, execute, observe)
    attackers = set(filter(None, args.attackers.split(",")))

    pkts: "queue.Queue" = queue.Queue()
    sniffer = AsyncSniffer(iface=args.iface, prn=pkts.put, store=False)
    sniffer.start()
    table, handled = FlowTable(), {}
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    end = time.time() + args.duration
    with open(args.log, "w") as log:
        while time.time() < end:
            try:
                info = scapy_to_packetinfo(pkts.get(timeout=0.5))
            except queue.Empty:
                finished = table.expire(time.time())
            else:
                finished = table.add(info) if info is not None else []
            for fl in finished:
                if fl.src == args.victim:                      # only inbound flows
                    continue
                if time.time() - handled.get(fl.src, -1e9) < args.cooldown:
                    continue
                t0 = time.perf_counter()
                x = fl.features()
                feat_ms = (time.perf_counter() - t0) * 1000.0
                res = agent.process(x, fl.src)
                if res.outcome in ("mitigated", "escalated", "failed"):
                    handled[fl.src] = time.time()
                row = asdict(res)
                row.update(agent=args.agent, scenario=args.scenario, src=fl.src,
                           gt="attack" if fl.src in attackers else "benign",
                           feature_ms=feat_ms, time=time.time())
                if res.latency_first_action_ms is not None:
                    row["latency_ms"] = feat_ms + res.latency_first_action_ms
                log.write(json.dumps(row) + "\n")
                log.flush()
    sniffer.stop()
    print(f"done; phi = {agent.phi:.3f}; log written to {args.log}")
    if args.execute:
        print("Remember to remove the iptables rules: sudo iptables -F INPUT && sudo iptables -F OUTPUT")


if __name__ == "__main__":
    main()

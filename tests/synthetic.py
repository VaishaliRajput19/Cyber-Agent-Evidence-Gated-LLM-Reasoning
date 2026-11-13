"""Synthetic traffic generator used ONLY for testing the pipeline offline.
Results on this data are not experimental results."""
import random

from cyber_agent.features import PacketInfo, flows_from_packets

VICTIM = "10.0.0.2"


def gen(label, src, t0, n_flows, rng):
    pk = []
    t = t0
    for i in range(n_flows):
        sport = rng.randint(30000, 60000)
        if label == "benign":
            dport = rng.choice([80, 443, 53])
            for k in range(rng.randint(6, 20)):
                fwd = k % 2 == 0
                t += rng.uniform(0.01, 0.2)
                pk.append(PacketInfo(t, src if fwd else VICTIM, VICTIM if fwd else src,
                                     sport if fwd else dport, dport if fwd else sport,
                                     "tcp", rng.randint(60, 1400), "A" if k else "S"))
            pk.append(PacketInfo(t + 0.01, src, VICTIM, sport, dport, "tcp", 60, "FA"))
        elif label == "port_scan":
            dport = rng.randint(1, 1024)
            t += 0.005
            pk.append(PacketInfo(t, src, VICTIM, sport, dport, "tcp", 60, "S"))
            pk.append(PacketInfo(t + 0.001, VICTIM, src, dport, sport, "tcp", 60, "R"))
        elif label == "ssh_bruteforce":
            for k in range(rng.randint(8, 12)):
                fwd = k % 2 == 0
                t += rng.uniform(0.005, 0.03)
                pk.append(PacketInfo(t, src if fwd else VICTIM, VICTIM if fwd else src,
                                     sport if fwd else 22, 22 if fwd else sport,
                                     "tcp", rng.randint(80, 200), "PA"))
            pk.append(PacketInfo(t + 0.01, VICTIM, src, 22, sport, "tcp", 60, "FA"))
        elif label == "ddos":
            for k in range(rng.randint(40, 80)):
                t += 0.0005
                pk.append(PacketInfo(t, src, VICTIM, sport, 80, "udp", rng.randint(500, 1400)))
        t += rng.uniform(0.01, 0.1)
    return pk, t


def dataset(seed=0):
    rng = random.Random(seed)
    rows = []
    for label, src, n in [("benign", "10.0.0.10", 400), ("benign", "10.0.0.11", 400),
                          ("port_scan", "10.0.0.66", 300), ("ssh_bruteforce", "10.0.0.67", 300),
                          ("ddos", "10.0.0.68", 200)]:
        pk, _ = gen(label, src, 0.0, n, rng)
        for fl in flows_from_packets(sorted(pk, key=lambda p: p.ts), idle_timeout=0.5):
            rows.append((fl.features(), label if fl.src == src and label != "benign" else "benign"))
    return rows

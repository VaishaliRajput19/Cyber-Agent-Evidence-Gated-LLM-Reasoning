"""Flow building and feature extraction (Eq. 1 of the paper).

Packets are grouped into bidirectional flows keyed by the 5-tuple. A flow is
emitted when it has been idle for `idle_timeout` seconds, has been active for
`active_timeout` seconds, or (for TCP) sees FIN/RST. Each flow is turned into
the feature vector x_i = [x_i1, ..., x_id] defined by FEATURE_NAMES.

This module does not depend on Scapy: capture code converts packets into
`PacketInfo` records, so the same logic works for live traffic and pcaps.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

FEATURE_NAMES = [
    "duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
    "mean_pkt_len", "std_pkt_len", "mean_iat", "std_iat",
    "pkts_per_s", "bytes_per_s",
    "syn_count", "ack_count", "fin_count", "rst_count",
    "dst_port", "proto",
    "src_unique_dst_ports", "src_flow_rate",
]

PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1}


@dataclass
class PacketInfo:
    ts: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: str            # "tcp" | "udp" | "icmp"
    length: int
    flags: str = ""       # TCP flags, e.g. "S", "SA", "FA", "R"


@dataclass
class Flow:
    src: str              # initiator
    dst: str
    sport: int
    dport: int
    proto: str
    start: float
    last: float
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    lengths: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    syn: int = 0
    ack: int = 0
    fin: int = 0
    rst: int = 0
    closed: bool = False
    src_unique_dst_ports: int = 0
    src_flow_rate: float = 0.0

    def add(self, p: PacketInfo, forward: bool) -> None:
        self.last = p.ts
        self.lengths.append(p.length)
        self.times.append(p.ts)
        if forward:
            self.fwd_pkts += 1
            self.fwd_bytes += p.length
        else:
            self.bwd_pkts += 1
            self.bwd_bytes += p.length
        if p.proto == "tcp":
            f = p.flags
            self.syn += "S" in f
            self.ack += "A" in f
            self.fin += "F" in f
            self.rst += "R" in f
            if "F" in f or "R" in f:
                self.closed = True

    def features(self) -> List[float]:
        n = len(self.lengths)
        dur = max(self.last - self.start, 1e-6)
        mean_len = sum(self.lengths) / n
        std_len = math.sqrt(sum((x - mean_len) ** 2 for x in self.lengths) / n)
        iats = [b - a for a, b in zip(self.times, self.times[1:])] or [0.0]
        mean_iat = sum(iats) / len(iats)
        std_iat = math.sqrt(sum((x - mean_iat) ** 2 for x in iats) / len(iats))
        total_bytes = self.fwd_bytes + self.bwd_bytes
        return [
            dur, self.fwd_pkts, self.bwd_pkts, self.fwd_bytes, self.bwd_bytes,
            mean_len, std_len, mean_iat, std_iat,
            n / dur, total_bytes / dur,
            self.syn, self.ack, self.fin, self.rst,
            self.dport, PROTO_NUM.get(self.proto, 0),
            self.src_unique_dst_ports, self.src_flow_rate,
        ]


class FlowTable:
    """Aggregates packets into flows and emits finished flows."""

    def __init__(self, idle_timeout: float = 2.0, active_timeout: float = 10.0,
                 host_window: float = 10.0) -> None:
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.host_window = host_window
        self.flows: Dict[Tuple, Flow] = {}
        # per-source history of (time, dst_port) for host-level features
        self.history: Dict[str, Deque[Tuple[float, int]]] = defaultdict(deque)

    @staticmethod
    def _key(p: PacketInfo) -> Tuple[Tuple, bool]:
        a = (p.src, p.sport)
        b = (p.dst, p.dport)
        if a <= b:
            return (a, b, p.proto), True
        return (b, a, p.proto), False

    def _host_features(self, src: str, now: float, dport: int) -> Tuple[int, float]:
        h = self.history[src]
        h.append((now, dport))
        while h and now - h[0][0] > self.host_window:
            h.popleft()
        return len({d for _, d in h}), len(h) / self.host_window

    def add(self, p: PacketInfo) -> List[Flow]:
        """Add one packet; return flows that finished as a result."""
        done = self.expire(p.ts)
        key, _ = self._key(p)
        flow = self.flows.get(key)
        if flow is None:
            flow = Flow(src=p.src, dst=p.dst, sport=p.sport, dport=p.dport,
                        proto=p.proto, start=p.ts, last=p.ts)
            flow.src_unique_dst_ports, flow.src_flow_rate = self._host_features(p.src, p.ts, p.dport)
            self.flows[key] = flow
        flow.add(p, forward=(p.src == flow.src and p.sport == flow.sport))
        if flow.closed:
            done.append(self.flows.pop(key))
        return done

    def expire(self, now: float) -> List[Flow]:
        out = []
        for k in list(self.flows):
            f = self.flows[k]
            if now - f.last > self.idle_timeout or now - f.start > self.active_timeout:
                out.append(self.flows.pop(k))
        return out

    def flush(self) -> List[Flow]:
        out = list(self.flows.values())
        self.flows.clear()
        return out


def flows_from_packets(packets: Iterable[PacketInfo], **kw) -> List[Flow]:
    table = FlowTable(**kw)
    out: List[Flow] = []
    for p in packets:
        out.extend(table.add(p))
    out.extend(table.flush())
    return out


def scapy_to_packetinfo(pkt) -> Optional[PacketInfo]:
    """Convert a Scapy packet to PacketInfo (returns None for non-IP)."""
    from scapy.layers.inet import IP, TCP, UDP, ICMP  # imported lazily
    if IP not in pkt:
        return None
    ip = pkt[IP]
    if TCP in pkt:
        t = pkt[TCP]
        return PacketInfo(float(pkt.time), ip.src, ip.dst, int(t.sport), int(t.dport),
                          "tcp", len(pkt), str(t.flags))
    if UDP in pkt:
        u = pkt[UDP]
        return PacketInfo(float(pkt.time), ip.src, ip.dst, int(u.sport), int(u.dport),
                          "udp", len(pkt))
    if ICMP in pkt:
        return PacketInfo(float(pkt.time), ip.src, ip.dst, 0, 0, "icmp", len(pkt))
    return None

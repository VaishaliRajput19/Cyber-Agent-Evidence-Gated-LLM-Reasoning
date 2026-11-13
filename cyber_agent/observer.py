"""Traffic observer used to verify mitigations (residual ratio rho, Eq. 7).

After an action against a source, the observer sniffs traffic from that
source for delta_t seconds, builds flows, and returns the fused label of each
flow. A flow counts as a *residual attack* only if it is flagged malicious
AND the victim still responds to it (bwd_pkts > 0): packets dropped by
iptables are still visible to the sniffer, so seeing attacker packets alone
does not mean the mitigation failed.
"""
from __future__ import annotations

from typing import List

from .features import FlowTable, scapy_to_packetinfo


class LiveObserver:
    def __init__(self, iface: str, detector, idle_timeout: float = 1.0) -> None:
        self.iface = iface
        self.detector = detector
        self.idle_timeout = idle_timeout

    def __call__(self, source: str, delta_t: float) -> List[int]:
        from scapy.all import sniff  # lazy import, needs root
        table = FlowTable(idle_timeout=self.idle_timeout)
        flows = []
        for pkt in sniff(iface=self.iface, filter=f"host {source}", timeout=delta_t, store=True):
            info = scapy_to_packetinfo(pkt)
            if info is not None:
                flows.extend(table.add(info))
        flows.extend(table.flush())
        labels = []
        for fl in flows:
            if fl.src != source:
                continue
            malicious = self.detector.fused(fl.features())
            labels.append(int(malicious and fl.bwd_pkts > 0))
        return labels

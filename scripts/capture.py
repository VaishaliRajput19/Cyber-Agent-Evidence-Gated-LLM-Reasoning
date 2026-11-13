"""Capture traffic (live or from a pcap) and write labeled flow features to CSV.

Run on the Ubuntu Server victim (root needed for live capture), once per scenario:
  sudo python scripts/capture.py --iface enp0s3 --label benign --duration 600 --out data/flows/benign.csv
  sudo python scripts/capture.py --iface enp0s3 --label port_scan --attacker 192.168.56.101 \
       --duration 300 --out data/flows/port_scan.csv
Flows whose initiator is the attacker get the scenario label; all others are "benign".
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cyber_agent.features import FEATURE_NAMES, FlowTable, scapy_to_packetinfo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface")
    ap.add_argument("--pcap", help="read packets from a pcap instead of live capture")
    ap.add_argument("--label", required=True, help="benign | recon | port_scan | ssh_bruteforce | ddos")
    ap.add_argument("--attacker", default=None, help="attacker IP (Kali); flows from it get --label")
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from scapy.all import rdpcap, sniff
    packets = rdpcap(args.pcap) if args.pcap else sniff(iface=args.iface, timeout=args.duration)

    table, flows = FlowTable(), []
    for pkt in packets:
        info = scapy_to_packetinfo(pkt)
        if info is not None:
            flows.extend(table.add(info))
    flows.extend(table.flush())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"] + FEATURE_NAMES + ["label"])
        for fl in flows:
            lab = args.label if (args.attacker is None or fl.src == args.attacker) else "benign"
            w.writerow([fl.src, fl.dst] + fl.features() + [lab])
    print(f"wrote {len(flows)} flows to {args.out}")


if __name__ == "__main__":
    main()

"""Action engine: maps allowlisted actions to iptables commands on the victim.

Runs in dry-run mode by default (prints commands). Set dry_run=False on the
Ubuntu Server victim, with root privileges, to execute them.
"""
import subprocess

COMMANDS = {
    "block_source_ip":   "iptables -A INPUT -s {src} -j DROP",
    "rate_limit_source": "iptables -A INPUT -s {src} -m limit --limit 10/second -j ACCEPT",
    "drop_port":         "iptables -A INPUT -s {src} -p tcp --dport {port} -j DROP",
    "isolate_host":      "iptables -I INPUT -s {src} -j DROP && iptables -I OUTPUT -d {src} -j DROP",
}


def make_executor(dry_run: bool = True, port: int = 22):
    def execute(action: str, source: str) -> bool:
        if action not in COMMANDS:
            return False
        cmd = COMMANDS[action].format(src=source, port=port)
        if dry_run:
            print(f"[dry-run] {cmd}")
            return True
        return subprocess.run(cmd, shell=True).returncode == 0

    return execute

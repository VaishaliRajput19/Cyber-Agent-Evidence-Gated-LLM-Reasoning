"""Re-implemented comparison approaches (Table I of the paper).

All agents share the detectors, knowledge base, LLM, action allowlist,
executor and observer with EGAR, so only the decision logic differs. Each
baseline acts once; the residual ratio rho is measured after the action only
to score mitigation success, never to change the decision.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from .egar import (BENIGN, FAST, NO_ACTION, REASON, EGARConfig, EGARResult,
                   ScoreNormalizer, evidence_label, normalized_entropy)

DEFAULT_PLAYBOOK = {
    "recon": "rate_limit_source",
    "port_scan": "block_source_ip",
    "ssh_bruteforce": "drop_port",
    "ddos": "rate_limit_source",
    "unknown-anomalous": "block_source_ip",
}


@dataclass
class _Base:
    cfg: EGARConfig
    categories: Sequence[str]
    f_b: Callable
    f_m: Callable
    f_if: Callable
    normalizer: ScoreNormalizer
    retrieve: Callable
    llm: Callable
    execute: Callable
    observe: Callable
    alerts: int = 0
    llm_alerts: int = 0

    @property
    def phi(self) -> float:
        return self.llm_alerts / self.alerts if self.alerts else 0.0

    def _evidence(self, x):
        p, q = float(self.f_b(x)), list(self.f_m(x))
        a = self.normalizer(float(self.f_if(x)))
        c = evidence_label(p, a, q, self.categories, self.cfg)
        return p, q, a, c

    def _is_alert(self, p, a) -> bool:          # union rule, Eq. (4)
        return p >= 0.5 or a > self.cfg.tau_tilde

    def _ask_llm(self, c, p, a, q, x) -> str:
        self.llm_alerts += 1
        act = self.llm(c=c, p=p, a=a, H=normalized_entropy(q), context=self.retrieve(c, x))
        return act if act in self.cfg.allowlist else NO_ACTION

    def _act_once(self, action, source, path, c, t0, llm_calls) -> EGARResult:
        if action == NO_ACTION:
            return EGARResult(path, c, NO_ACTION, "no_action", llm_calls=llm_calls)
        self.execute(action, source)
        lat = (time.perf_counter() - t0) * 1000.0
        labels = self.observe(source, self.cfg.delta_t)
        rho = sum(labels) / len(labels) if labels else 0.0
        outcome = "mitigated" if rho <= self.cfg.rho0 else "failed"
        return EGARResult(path, c, action, outcome, rounds=1, rho=rho,
                          llm_calls=llm_calls, latency_first_action_ms=lat)


@dataclass
class StaticPlaybook(_Base):
    """(i) SOAR-style fixed action per category, no LLM."""
    playbook: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PLAYBOOK))

    def process(self, x, source) -> EGARResult:
        t0 = time.perf_counter()
        p, q, a, c = self._evidence(x)
        if not self._is_alert(p, a):
            return EGARResult(BENIGN, c, NO_ACTION, "no_action")
        self.alerts += 1
        return self._act_once(self.playbook.get(c, "block_source_ip"), source, FAST, c, t0, 0)


@dataclass
class LLMAlways(_Base):
    """(ii) IDS-Agent-style: LLM reasons over every alert, action executed once."""

    def process(self, x, source) -> EGARResult:
        t0 = time.perf_counter()
        p, q, a, c = self._evidence(x)
        if not self._is_alert(p, a):
            return EGARResult(BENIGN, c, NO_ACTION, "no_action")
        self.alerts += 1
        return self._act_once(self._ask_llm(c, p, a, q, x), source, REASON, c, t0, 1)


@dataclass
class MemoryAgent(_Base):
    """(iii) Memory-augmented agent: reuses the first LLM decision per category
    without verifying whether it worked."""
    memory: Dict[str, str] = field(default_factory=dict)

    def process(self, x, source) -> EGARResult:
        t0 = time.perf_counter()
        p, q, a, c = self._evidence(x)
        if not self._is_alert(p, a):
            return EGARResult(BENIGN, c, NO_ACTION, "no_action")
        self.alerts += 1
        if c in self.memory:
            return self._act_once(self.memory[c], source, FAST, c, t0, 0)
        action = self._ask_llm(c, p, a, q, x)
        if action != NO_ACTION:
            self.memory[c] = action
        return self._act_once(action, source, REASON, c, t0, 1)


@dataclass
class ConfidenceGating(_Base):
    """(iv) Invoke the LLM only when p_i < theta_h; otherwise use the static
    playbook. No uncertainty/disagreement routing and no verification."""
    playbook: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PLAYBOOK))

    def process(self, x, source) -> EGARResult:
        t0 = time.perf_counter()
        p, q, a, c = self._evidence(x)
        if not self._is_alert(p, a):
            return EGARResult(BENIGN, c, NO_ACTION, "no_action")
        self.alerts += 1
        if p >= self.cfg.theta_h:
            return self._act_once(self.playbook.get(c, "block_source_ip"), source, FAST, c, t0, 0)
        return self._act_once(self._ask_llm(c, p, a, q, x), source, REASON, c, t0, 1)


AGENTS = {"static": StaticPlaybook, "llm_always": LLMAlways,
          "memory": MemoryAgent, "confidence": ConfidenceGating}

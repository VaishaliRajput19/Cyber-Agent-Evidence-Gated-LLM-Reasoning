"""EGAR: Evidence-Gated Adaptive Reasoning and Response.

Reference implementation of Algorithm 1 in the Cyber Agent paper.

EGAR decides, from the evidence produced by the hybrid detection layer,
whether a flow is benign, can be handled by a cached playbook action (FAST),
or needs retrieval-grounded LLM reasoning (REASON). After an action is
executed, EGAR verifies its effect with the residual attack ratio rho,
re-plans up to R rounds if the attack persists, and distills actions that
were verified m times into the playbook cache.

The detectors, retriever, LLM, action executor and traffic observer are
injected as callables, so this module has no hard dependency on Scapy,
XGBoost, scikit-learn or Ollama and can be unit-tested with mocks.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

BENIGN, FAST, REASON = "BENIGN", "FAST", "REASON"
UNKNOWN_ANOMALOUS = "unknown-anomalous"
NO_ACTION = "no_action"


@dataclass
class EGARConfig:
    """Thresholds and limits of EGAR (symbols as in the paper)."""

    theta_l: float = 0.5   # low confidence threshold on p_i
    theta_h: float = 0.9   # high confidence threshold on p_i (tune on validation)
    eta: float = 0.3       # max normalized category uncertainty H_i
    epsilon: float = 0.3   # max disagreement delta_i = |p_i - a_i|
    tau_tilde: float = 0.5 # normalized Isolation Forest threshold
    rho0: float = 0.1      # tolerated residual attack ratio
    R: int = 2             # max verify/re-plan rounds
    m: int = 3             # verified successes needed for distillation
    delta_t: float = 5.0   # observation window after an action (seconds)
    allowlist: Sequence[str] = (
        "block_source_ip",
        "rate_limit_source",
        "drop_port",
        "isolate_host",
    )


class ScoreNormalizer:
    """Min-max normalization of Isolation Forest anomaly scores.

    Uses the convention of the paper: higher s_i = more anomalous.
    For scikit-learn, pass s = -IsolationForest.score_samples(X).
    """

    def __init__(self) -> None:
        self.s_min: Optional[float] = None
        self.s_max: Optional[float] = None

    def fit(self, train_scores: Sequence[float]) -> "ScoreNormalizer":
        self.s_min, self.s_max = float(min(train_scores)), float(max(train_scores))
        return self

    def __call__(self, s: float) -> float:
        if self.s_min is None or self.s_max is None:
            raise RuntimeError("ScoreNormalizer must be fitted first")
        span = max(self.s_max - self.s_min, 1e-12)
        return min(max((s - self.s_min) / span, 0.0), 1.0)


def normalized_entropy(q: Sequence[float]) -> float:
    """H_i = -(1/log K) * sum_k q_ik log q_ik, in [0, 1]."""
    K = len(q)
    if K <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in q if p > 0)
    return h / math.log(K)


def route(p: float, a: float, H: float, delta: float, cfg: EGARConfig) -> str:
    """Routing rule g_i of Eq. (6)."""
    if p < cfg.theta_l and a < cfg.tau_tilde:
        return BENIGN
    if p >= cfg.theta_h and H <= cfg.eta and delta <= cfg.epsilon:
        return FAST
    return REASON


def evidence_label(p: float, a: float, q: Sequence[float],
                   categories: Sequence[str], cfg: EGARConfig) -> str:
    """Evidence label c_i of Eq. (5)."""
    if p >= 0.5:
        return categories[max(range(len(q)), key=lambda k: q[k])]
    if a > cfg.tau_tilde:
        return UNKNOWN_ANOMALOUS
    return "benign"


@dataclass
class EGARResult:
    path: str                 # BENIGN, FAST or REASON
    label: str                # evidence label c_i
    action: str               # final action (or no_action)
    outcome: str              # no_action | mitigated | escalated
    rounds: int = 0
    rho: Optional[float] = None
    llm_calls: int = 0
    latency_first_action_ms: Optional[float] = None


@dataclass
class EGAR:
    """EGAR controller (Algorithm 1)."""

    cfg: EGARConfig
    categories: Sequence[str]
    f_b: Callable[[Sequence[float]], float]               # -> p_i
    f_m: Callable[[Sequence[float]], Sequence[float]]     # -> q_i
    f_if: Callable[[Sequence[float]], float]              # -> s_i (higher = anomalous)
    normalizer: ScoreNormalizer
    retrieve: Callable[[str, Sequence[float]], str]       # (c_i, x_i) -> context C
    llm: Callable[..., str]                               # -> action from allowlist or no_action
    execute: Callable[[str, str], bool]                   # (action, source) -> executed?
    observe: Callable[[str, float], List[int]]            # (source, delta_t) -> fused labels y^(f)_j
    playbook: Dict[str, str] = field(default_factory=dict)
    successes: Dict[tuple, int] = field(default_factory=lambda: defaultdict(int))
    alerts: int = 0
    llm_alerts: int = 0

    # ------------------------------------------------------------------
    def _validate(self, action: str, tried: Sequence[str]) -> str:
        """Constrain the LLM output to the allowlist (or no_action)."""
        if action == NO_ACTION:
            return NO_ACTION
        if action in self.cfg.allowlist and action not in tried:
            return action
        for a in self.cfg.allowlist:          # fall back to an untried allowed action
            if a not in tried:
                return a
        return NO_ACTION

    def residual_ratio(self, labels: List[int]) -> float:
        """Residual attack ratio rho of Eq. (7)."""
        return sum(labels) / len(labels) if labels else 0.0

    @property
    def phi(self) -> float:
        """Fraction of non-benign alerts routed to the LLM."""
        return self.llm_alerts / self.alerts if self.alerts else 0.0

    # ------------------------------------------------------------------
    def process(self, x: Sequence[float], source: str) -> EGARResult:
        t0 = time.perf_counter()
        p = float(self.f_b(x))
        q = list(self.f_m(x))
        a = self.normalizer(float(self.f_if(x)))
        H = normalized_entropy(q)
        delta = abs(p - a)
        c = evidence_label(p, a, q, self.categories, self.cfg)
        g = route(p, a, H, delta, self.cfg)

        if g == BENIGN:
            return EGARResult(path=g, label=c, action=NO_ACTION, outcome="no_action")

        self.alerts += 1
        llm_calls, context = 0, None
        if g == FAST and c in self.playbook:
            action = self.playbook[c]
            from_llm = False
        else:
            self.llm_alerts += 1
            context = self.retrieve(c, x)
            action = self._validate(self.llm(c=c, p=p, a=a, H=H, context=context), [])
            llm_calls += 1
            from_llm = True
            if action == NO_ACTION:
                return EGARResult(path=g, label=c, action=NO_ACTION,
                                  outcome="no_action", llm_calls=llm_calls)

        tried: List[str] = []
        latency_ms, rho = None, None
        for r in range(1, self.cfg.R + 1):
            self.execute(action, source)
            if latency_ms is None:
                latency_ms = (time.perf_counter() - t0) * 1000.0
            tried.append(action)
            rho = self.residual_ratio(self.observe(source, self.cfg.delta_t))
            if rho <= self.cfg.rho0:
                if from_llm:
                    self.successes[(c, action)] += 1
                    if self.successes[(c, action)] >= self.cfg.m:
                        self.playbook[c] = action          # experience distillation
                return EGARResult(path=g, label=c, action=action, outcome="mitigated",
                                  rounds=r, rho=rho, llm_calls=llm_calls,
                                  latency_first_action_ms=latency_ms)
            if r == self.cfg.R:
                break
            if context is None:
                context = self.retrieve(c, x)
            action = self._validate(
                self.llm(c=c, p=p, a=a, H=H, context=context, feedback=(action, rho)), tried)
            llm_calls += 1
            from_llm = True
            if action == NO_ACTION:
                break

        return EGARResult(path=g, label=c, action=tried[-1], outcome="escalated",
                          rounds=len(tried), rho=rho, llm_calls=llm_calls,
                          latency_first_action_ms=latency_ms)

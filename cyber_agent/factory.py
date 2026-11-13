"""Builds EGAR or a baseline agent from a config file and trained models."""
from __future__ import annotations

import yaml

from .baselines import AGENTS
from .egar import EGAR, EGARConfig

CFG_KEYS = {"theta_l", "theta_h", "eta", "epsilon", "rho0", "R", "m", "delta_t", "allowlist"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_agent(kind: str, conf: dict, detector, kb, llm, execute, observe):
    cfg = EGARConfig(**{k: v for k, v in conf.items() if k in CFG_KEYS})
    cfg.tau_tilde = detector.tau_tilde            # Isolation Forest threshold, normalized
    common = dict(cfg=cfg, categories=detector.categories, f_b=detector.f_b,
                  f_m=detector.f_m, f_if=detector.f_if, normalizer=detector.normalizer,
                  retrieve=kb.retrieve, llm=llm, execute=execute, observe=observe)
    if kind == "egar":
        return EGAR(**common)
    if kind == "egar_r1":
        cfg.R = 1
        return EGAR(**common)
    return AGENTS[kind](**common)

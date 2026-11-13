"""Unit tests for EGAR with mock detectors (no Scapy/XGBoost/Ollama needed)."""
from cyber_agent import EGAR, EGARConfig, ScoreNormalizer, route, normalized_entropy

CATS = ["recon", "port_scan", "ssh_bruteforce"]


def make_agent(p, q, s, residuals, llm_actions):
    calls = {"llm": 0}
    res = iter(residuals)
    acts = iter(llm_actions)

    def llm(**kw):
        calls["llm"] += 1
        return next(acts)

    agent = EGAR(
        cfg=EGARConfig(),
        categories=CATS,
        f_b=lambda x: p, f_m=lambda x: q, f_if=lambda x: s,
        normalizer=ScoreNormalizer().fit([0.0, 1.0]),
        retrieve=lambda c, x: "kb context",
        llm=llm,
        execute=lambda a, src: True,
        observe=lambda src, dt: next(res),
    )
    return agent, calls


def test_entropy_bounds():
    assert normalized_entropy([1, 0, 0]) == 0.0
    assert abs(normalized_entropy([1/3, 1/3, 1/3]) - 1.0) < 1e-9


def test_benign_routing():
    cfg = EGARConfig()
    assert route(0.1, 0.1, 0.0, 0.0, cfg) == "BENIGN"


def test_unknown_anomalous_goes_to_llm():
    agent, calls = make_agent(0.1, [0.4, 0.3, 0.3], 0.9, [[0, 0, 0]], ["block_source_ip"])
    r = agent.process([0.0], "10.0.0.5")
    assert r.path == "REASON" and r.label == "unknown-anomalous"
    assert r.outcome == "mitigated" and calls["llm"] == 1


def test_replan_when_attack_persists():
    agent, calls = make_agent(0.1, [0.4, 0.3, 0.3], 0.9,
                              [[1, 1, 1], [0, 0, 0]], ["rate_limit_source", "block_source_ip"])
    r = agent.process([0.0], "10.0.0.5")
    assert r.outcome == "mitigated" and r.rounds == 2 and calls["llm"] == 2


def test_distillation_then_fast_path():
    # confident, known, agreeing evidence -> FAST, but cache empty at first
    agent, calls = make_agent(0.98, [0.98, 0.01, 0.01], 0.95,
                              [[0]] * 4, ["block_source_ip"] * 3)
    for _ in range(3):                      # m = 3 verified successes
        agent.process([0.0], "10.0.0.7")
    assert agent.playbook["recon"] == "block_source_ip"
    r = agent.process([0.0], "10.0.0.7")    # now served from the playbook
    assert r.path == "FAST" and r.llm_calls == 0 and calls["llm"] == 3

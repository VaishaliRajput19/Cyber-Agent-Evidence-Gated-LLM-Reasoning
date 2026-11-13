"""LLM adapter: Qwen 2.5 (3B) served locally through Ollama.

Returns one action from the allowlist, or "no_action".
Requires: `pip install ollama` and `ollama pull qwen2.5:3b`.
"""
from typing import Optional, Sequence, Tuple

PROMPT = """You are a network security agent.
Evidence: category={c}, malicious probability={p:.2f}, anomaly score={a:.2f}, category uncertainty={H:.2f}.
Retrieved knowledge:
{context}
{feedback}
Choose exactly ONE action from this list, or "no_action" if the traffic is benign:
{allowlist}
Answer with the action name only."""


def make_llm(allowlist: Sequence[str], model: str = "qwen2.5:3b"):
    import ollama

    def llm(c: str, p: float, a: float, H: float, context: str,
            feedback: Optional[Tuple[str, float]] = None) -> str:
        fb = ""
        if feedback:
            fb = (f"Previous action '{feedback[0]}' was executed but the attack persists "
                  f"(residual ratio {feedback[1]:.2f}). Choose a different, stronger action.")
        msg = PROMPT.format(c=c, p=p, a=a, H=H, context=context, feedback=fb,
                            allowlist=", ".join(list(allowlist) + ["no_action"]))
        out = ollama.chat(model=model, messages=[{"role": "user", "content": msg}])
        text = out["message"]["content"].strip().lower()
        for act in list(allowlist) + ["no_action"]:
            if act in text:
                return act
        return "no_action"

    return llm

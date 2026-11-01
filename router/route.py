"""Cost/latency-aware model routing.

The core idea: you do not need GPT-4-class reasoning to answer "what is the
capital of France". Routing cheap requests to a cheap model is the single
largest cost lever in an LLM product — often 60-80% of spend.

Routing is deterministic and explainable. A black-box router that silently
sends a hard question to a weak model is worse than no router at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Prices in USD per 1M tokens. Verified against published provider pricing
# pages at time of writing; treat as configuration, not gospel — providers
# change these and you should re-check before relying on the numbers.
MODELS: Dict[str, dict] = {
    "local/llama3.1-8b": {
        "input": 0.0, "output": 0.0, "tier": 0,
        "latency_ms": 400, "context": 128_000, "quality": 0.55,
    },
    "groq/llama-3.1-8b-instant": {
        "input": 0.05, "output": 0.08, "tier": 1,
        "latency_ms": 250, "context": 128_000, "quality": 0.60,
    },
    "gpt-4o-mini": {
        "input": 0.15, "output": 0.60, "tier": 2,
        "latency_ms": 900, "context": 128_000, "quality": 0.75,
    },
    "claude-haiku": {
        "input": 0.80, "output": 4.00, "tier": 2,
        "latency_ms": 1100, "context": 200_000, "quality": 0.78,
    },
    "gpt-4o": {
        "input": 2.50, "output": 10.00, "tier": 3,
        "latency_ms": 1800, "context": 128_000, "quality": 0.92,
    },
    "claude-sonnet": {
        "input": 3.00, "output": 15.00, "tier": 3,
        "latency_ms": 2000, "context": 200_000, "quality": 0.94,
    },
}

# Signals that a request genuinely needs a strong model.
_HARD_SIGNALS = [
    (r"\b(prove|derive|theorem|formal(ly)? verify)\b", 3),
    (r"\b(refactor|architect\w*|design (a|the) system)\b", 2),
    (r"\b(debug|root cause|why does this fail)\b", 2),
    (r"\b(security|vulnerab|exploit|threat model)\b", 2),
    (r"\b(step[- ]by[- ]step|chain of thought|reason carefully)\b", 2),
    (r"\b(write (a )?(full|complete|production))\b", 2),
    (r"\b(legal|medical|financial advice|contract)\b", 3),
]
_EASY_SIGNALS = [
    (r"^(hi|hello|hey|thanks|thank you|ok|okay)\b", -3),
    (r"\b(what is|who is|define|translate|summarise|summarize)\b", -1),
    (r"\b(format|convert|list|extract|classify)\b", -1),
]
_HARD_RE = [(re.compile(p, re.I), w) for p, w in _HARD_SIGNALS]
_EASY_RE = [(re.compile(p, re.I), w) for p, w in _EASY_SIGNALS]

CODE_FENCE = re.compile(r"```")


@dataclass
class Request:
    prompt: str
    max_output_tokens: int = 500
    require_json: bool = False
    max_latency_ms: Optional[int] = None
    min_quality: float = 0.0
    budget_usd: Optional[float] = None


@dataclass
class Decision:
    model: str
    complexity: int
    reason: str
    est_cost_usd: float
    est_latency_ms: int
    alternatives: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model": self.model, "complexity": self.complexity,
            "reason": self.reason,
            "est_cost_usd": round(self.est_cost_usd, 6),
            "est_latency_ms": self.est_latency_ms,
            "alternatives": self.alternatives,
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for English, ~3 for code.

    Deliberately not a real tokenizer — this runs on every request and a
    tokenizer dependency would cost more than the accuracy is worth here.
    """
    if not text:
        return 0
    per = 3 if CODE_FENCE.search(text) else 4
    return max(1, len(text) // per)


def complexity_score(req: Request) -> tuple[int, List[str]]:
    """0-10. Higher means it needs a stronger model."""
    text = req.prompt or ""
    score = 3
    reasons: List[str] = []

    for rx, w in _HARD_RE:
        if rx.search(text):
            score += w
            reasons.append(f"hard signal: {rx.pattern[:28]}")

    for rx, w in _EASY_RE:
        if rx.search(text):
            score += w
            reasons.append(f"easy signal: {rx.pattern[:28]}")

    tokens = estimate_tokens(text)
    if tokens > 2000:
        score += 2
        reasons.append(f"long input ({tokens} tok)")
    elif tokens > 600:
        score += 1
        reasons.append(f"medium input ({tokens} tok)")

    if CODE_FENCE.search(text):
        score += 1
        reasons.append("contains code")

    if req.require_json:
        score += 1
        reasons.append("structured output required")

    if req.max_output_tokens > 1500:
        score += 1
        reasons.append("long output requested")

    return max(0, min(10, score)), reasons


def cost_of(model: str, in_tokens: int, out_tokens: int) -> float:
    m = MODELS[model]
    return (in_tokens / 1_000_000) * m["input"] + (out_tokens / 1_000_000) * m["output"]


def route(req: Request) -> Decision:
    score, reasons = complexity_score(req)
    in_tok = estimate_tokens(req.prompt)
    out_tok = req.max_output_tokens

    # Minimum tier implied by complexity: 0-3 -> tier 1, 4-6 -> tier 2, 7+ -> tier 3
    min_tier = 1 if score <= 3 else 2 if score <= 6 else 3

    candidates = []
    for name, m in MODELS.items():
        if m["tier"] < min_tier:
            continue
        if m["quality"] < req.min_quality:
            continue
        if req.max_latency_ms and m["latency_ms"] > req.max_latency_ms:
            continue
        c = cost_of(name, in_tok, out_tok)
        if req.budget_usd is not None and c > req.budget_usd:
            continue
        candidates.append((c, m["latency_ms"], name, m))

    if not candidates:
        # Nothing satisfies the constraints — fall back to the strongest model
        # and say so, rather than silently returning a wrong answer.
        best = max(MODELS.items(), key=lambda kv: kv[1]["quality"])
        return Decision(
            model=best[0], complexity=score,
            reason="no model met the constraints; fell back to highest quality",
            est_cost_usd=cost_of(best[0], in_tok, out_tok),
            est_latency_ms=best[1]["latency_ms"],
        )

    # Cheapest that clears the bar; tie-break on latency.
    candidates.sort(key=lambda t: (t[0], t[1]))
    cost, latency, name, _ = candidates[0]

    alts = [
        {"model": n, "cost_usd": round(c, 6), "latency_ms": m["latency_ms"]}
        for c, _, n, m in candidates[1:4]
    ]

    return Decision(
        model=name, complexity=score,
        reason="; ".join(reasons) or "no strong signals; defaulted to cheapest capable model",
        est_cost_usd=cost, est_latency_ms=latency, alternatives=alts,
    )


def compare(req: Request) -> dict:
    """What routing saves versus always using the strongest model."""
    d = route(req)
    in_tok = estimate_tokens(req.prompt)
    strongest = max(MODELS.items(), key=lambda kv: kv[1]["quality"])[0]
    baseline = cost_of(strongest, in_tok, req.max_output_tokens)
    saved = baseline - d.est_cost_usd
    pct = (saved / baseline * 100) if baseline > 0 else 0.0
    return {
        "routed": d.as_dict(),
        "baseline_model": strongest,
        "baseline_cost_usd": round(baseline, 6),
        "saved_usd": round(saved, 6),
        "saved_pct": round(pct, 1),
    }

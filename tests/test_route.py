"""Routing tests — the behaviour that actually saves money."""
import pytest

from router.route import (
    MODELS,
    Request,
    compare,
    complexity_score,
    cost_of,
    estimate_tokens,
    route,
)


class TestTokenEstimate:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_english_roughly_four_chars(self):
        assert 20 <= estimate_tokens("a" * 100) <= 30

    def test_code_uses_denser_ratio(self):
        prose = estimate_tokens("word " * 100)
        code = estimate_tokens("```\n" + "word " * 100 + "\n```")
        assert code > prose, "code should estimate more tokens per character"


class TestComplexity:
    def test_greeting_is_trivial(self):
        s, _ = complexity_score(Request("hi"))
        assert s <= 2

    def test_simple_question_is_low(self):
        s, _ = complexity_score(Request("What is the capital of France?"))
        assert s <= 3

    def test_proof_is_high(self):
        s, _ = complexity_score(Request("Prove that this algorithm terminates."))
        assert s >= 6

    def test_security_question_escalates(self):
        s, _ = complexity_score(Request("Threat model this API for vulnerabilities."))
        assert s >= 5

    def test_long_input_escalates(self):
        short, _ = complexity_score(Request("summarise this"))
        long, _ = complexity_score(Request("summarise this " + "x " * 3000))
        assert long > short

    def test_json_requirement_escalates(self):
        a, _ = complexity_score(Request("extract names"))
        b, _ = complexity_score(Request("extract names", require_json=True))
        assert b > a

    def test_score_is_bounded(self):
        s, _ = complexity_score(Request("prove derive theorem " * 50))
        assert 0 <= s <= 10


class TestRouting:
    def test_trivial_request_goes_cheap(self):
        d = route(Request("hi"))
        assert MODELS[d.model]["tier"] <= 2
        assert d.est_cost_usd < 0.001

    def test_hard_request_goes_strong(self):
        d = route(Request("Prove the security properties of this protocol step by step."))
        assert MODELS[d.model]["tier"] == 3

    def test_never_picks_below_min_tier(self):
        d = route(Request("Prove a theorem about this system architecture."))
        assert MODELS[d.model]["tier"] >= 3

    def test_respects_min_quality(self):
        d = route(Request("hi", min_quality=0.9))
        assert MODELS[d.model]["quality"] >= 0.9

    def test_respects_latency_ceiling(self):
        d = route(Request("hi", max_latency_ms=300))
        assert MODELS[d.model]["latency_ms"] <= 300

    def test_respects_budget(self):
        d = route(Request("hi", budget_usd=0.000001))
        assert d.est_cost_usd <= 0.000001 or "fell back" in d.reason

    def test_impossible_constraints_fall_back_and_say_so(self):
        """Silently returning a weak model would be worse than a clear fallback."""
        d = route(Request("hi", min_quality=0.99, max_latency_ms=1))
        assert "fell back" in d.reason

    def test_decision_is_explainable(self):
        d = route(Request("Prove this theorem."))
        assert d.reason and d.complexity >= 0

    def test_alternatives_are_offered(self):
        d = route(Request("What is 2+2?"))
        assert isinstance(d.alternatives, list)


class TestCost:
    def test_local_is_free(self):
        assert cost_of("local/llama3.1-8b", 100_000, 100_000) == 0.0

    def test_cost_scales_with_tokens(self):
        a = cost_of("gpt-4o", 1000, 1000)
        b = cost_of("gpt-4o", 2000, 2000)
        assert b == pytest.approx(a * 2)

    def test_output_costs_more_than_input(self):
        assert cost_of("gpt-4o", 0, 1000) > cost_of("gpt-4o", 1000, 0)


class TestCompare:
    def test_simple_request_saves_most(self):
        r = compare(Request("What is the capital of France?"))
        assert r["saved_pct"] > 50, "routing a trivial question should save a lot"

    def test_hard_request_saves_little(self):
        r = compare(Request("Prove the security properties of this protocol step by step."))
        assert r["saved_pct"] < 50

    def test_saved_is_never_negative(self):
        for prompt in ["hi", "prove a theorem", "summarise this", "x " * 500]:
            assert compare(Request(prompt))["saved_usd"] >= 0

    def test_baseline_is_strongest_model(self):
        r = compare(Request("hi"))
        assert MODELS[r["baseline_model"]]["quality"] == max(
            m["quality"] for m in MODELS.values())

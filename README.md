# llm-router

[![tests](https://github.com/kkcyber2/llm-router/actions/workflows/tests.yml/badge.svg)](https://github.com/kkcyber2/llm-router/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Send each request to the cheapest model that can actually handle it, and show
the reasoning. No API key, no network — it is a pure decision function.

```bash
python -m router.cli "What is the capital of France?" --compare
```

```
model      groq/llama-3.1-8b-instant
complexity 2/10
reason     easy signal: ^(hi|hello|hey|thanks|thank you|ok|okay)\b; easy signal: \b(what is|who is|define|translate|summarise|summarize)\b
cost       $0.000040
baseline   $0.005150 (claude-sonnet)
saved      $0.005110 (99.2%)
```

```bash
python -m router.cli "Prove the security properties of this protocol step by step" --compare
# -> claude-sonnet, complexity 10/10, saved 0.0%
```

## Why routing is the biggest cost lever

Most production traffic is not hard. Greetings, lookups, formatting, extraction
and classification do not need a frontier model, and they are usually the
majority of requests. Routing them down is typically a 60-80% spend reduction
with no measurable quality loss on those requests.

## The design decision that matters

**Every decision is explainable.** `Decision.reason` names the exact signals
that fired. A router that silently sends a hard question to a weak model is
worse than no router — you would never find out until a customer did.

**Impossible constraints fall back loudly.** If nothing satisfies
`min_quality` + `max_latency_ms`, it returns the strongest model with
`"fell back"` in the reason rather than quietly picking something that fails
the constraint. `test_impossible_constraints_fall_back_and_say_so` pins it.

**Complexity is bounded 0-10 and tier-mapped.** 0-3 → tier 1, 4-6 → tier 2,
7+ → tier 3. A request can never be routed below the tier its complexity
implies, so a hard prompt cannot be cheapened into a wrong answer.

## Signals

| Escalates | De-escalates |
|---|---|
| prove / derive / theorem / formally verify | greeting |
| refactor / architect / design a system | what is / who is / define / translate / summarise |
| debug / root cause | format / convert / list / extract / classify |
| security / vulnerability / exploit / threat model | |
| step-by-step / reason carefully | |
| legal / medical / financial advice | |
| long input, code fences, JSON required, long output | |

## Tests

```bash
python -m pytest -q     # 30 passed
```

## Limitations — read this

- **Prices are hardcoded and will drift.** They are configuration, not truth.
  Re-check provider pricing pages before trusting the cost numbers.
- **Token estimation is a heuristic** (~4 chars/token, ~3 for code). It is not a
  tokenizer; a real one would cost more per request than the accuracy is worth.
- **Signal matching is regex, not a classifier.** It will miss paraphrases and
  can be fooled by a hard question phrased simply. Treat it as a first pass.
- **No feedback loop.** It does not learn from which answers were actually good.
  A real system should log quality per route and retune the weights.
- **No streaming or fallback-on-error.** It decides; you execute.

## License

MIT — see [LICENSE](LICENSE).

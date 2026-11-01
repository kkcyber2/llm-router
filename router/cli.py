"""CLI: route a prompt and show the cost saving."""
from __future__ import annotations

import argparse
import json
import sys

from .route import Request, compare, route


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llm-router")
    ap.add_argument("prompt")
    ap.add_argument("--max-output", type=int, default=500)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="show saving versus always using the strongest model")
    args = ap.parse_args(argv)

    req = Request(args.prompt, max_output_tokens=args.max_output)
    result = compare(req) if args.compare else route(req).as_dict()

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    if args.compare:
        r = result["routed"]
        print(f"model      {r['model']}")
        print(f"complexity {r['complexity']}/10")
        print(f"reason     {r['reason']}")
        print(f"cost       ${r['est_cost_usd']:.6f}")
        print(f"baseline   ${result['baseline_cost_usd']:.6f} ({result['baseline_model']})")
        print(f"saved      ${result['saved_usd']:.6f} ({result['saved_pct']}%)")
    else:
        print(f"{result['model']}  (complexity {result['complexity']}/10, "
              f"${result['est_cost_usd']:.6f})")
        print(f"  {result['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

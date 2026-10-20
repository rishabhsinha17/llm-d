#!/usr/bin/env python3
"""Derive the prefill work-range boundary for remaining-work routing.

Remaining-work routing (`context-length-aware` with `reusableTokensProducerName`
in llm-d-router, see llm-d/llm-d-router#2559) splits the prefill pool into a
short-work and a long-work class by the `llm-d.ai/prefill-work-range` label and
routes a cached follow-up by the prefill work left after the pullable prefix.
This script places the boundary between the two classes from numbers the
calibration recipes in this directory already measure, so no further
in-cluster probe is needed:

- `peakPrefillThroughput` (tokens/s per rank) printed by `calibrate.sh`, or
  fitted from the recompute column of the pull ladder below;
- the pull-versus-recompute ladder printed by the
  `calibrate-min-cached-token-delta.sh` Job (`tokens recompute_ms pull_ms ...`
  rows), fitted to a pull-cost model, or an explicit `--pull-floor-s` and
  `--pull-per-token-us`.

Model
-----
A pull costs a fixed floor plus a per-token share; a recompute costs the
tokens divided by the peak prefill throughput of one rank:

    pull_time(t) = pull_floor + t * pull_per_token
    recompute(t) = t / peak_prefill_throughput

Migrating a follow-up to the short class only pays above the crossover where
the pull beats the recompute (economic anchor), and a request counts as long
work once its prefill occupies a rank for longer than the TTFT penalty the
load gate (`prefix-cache-affinity-filter`, `maxTTFTPenaltyMs`) tolerates for
the short class (SLO anchor):

    crossover  = pull_floor / (1 / throughput - pull_per_token)
    slo_tokens = throughput * ttft_penalty

The recommended boundary is the crossover rounded up to the KV block size,
provided it still fits under `slo_tokens`. If no block multiple fits in that
band the script says so and falls back to `slo_tokens`; if the crossover
itself exceeds `slo_tokens` the band is empty: pulls cannot pay within the
declared budget, so raise the budget or do not split the pool.

`minCachedTokenDelta` (D) is part of the routed work: the plugin routes a
follow-up on `newTokens + D - 1`, so the short-class ceiling is
`boundary + D - 1` and the long class starts one token above it (plugin
README, "P2P Cache-Aware Prefill Work").

Examples
--------
Explicit pull model, the GLM-5.2-FP8 on H200 reference (3585 tokens/s per
rank, pull floor 1.25 s plus 134 us per token, 3500 ms budget):

    derive_work_boundary.py --peak-prefill-throughput 3585 \
        --ttft-penalty-ms 3500 --block-size 64 --max-context 120000 \
        --pull-floor-s 1.25 --pull-per-token-us 134

Pull model fitted from the saved Job log of calibrate-min-cached-token-delta.sh:

    derive_work_boundary.py --peak-prefill-throughput 3585 \
        --ttft-penalty-ms 3500 --block-size 64 --max-context 120000 \
        --pull-ladder calibration-min-cached-token-delta.log

Stdlib only. Prints the anchors, the recommendation, and the label and EPP
parameter block to paste into the prefill pod templates and the router values.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass

EXIT_NO_RECOMMENDATION = 1

# One ladder row as printed by the calibration-min-cached-token-delta Job:
# "    8192        2286.4     2350.1    +2.8%       45.6"
LADDER_ROW = re.compile(
    r"^\s*(?P<tokens>\d+)\s+(?P<recompute_ms>\d+(?:\.\d+)?)\s+"
    r"(?P<pull_ms>\d+(?:\.\d+)?)\s+[+-]?\d+(?:\.\d+)?%\s+\d+(?:\.\d+)?\s*$"
)


@dataclass(frozen=True)
class LadderRow:
    tokens: int
    recompute_ms: float
    pull_ms: float


@dataclass(frozen=True)
class PullModel:
    floor_s: float
    per_token_us: float


@dataclass(frozen=True)
class Derivation:
    throughput: float
    per_token_compute_us: float
    pull: PullModel
    ttft_penalty_ms: float
    block_size: int
    max_context: int
    delta: int
    crossover: float | None  # None: pulling never beats recomputing
    slo_tokens: float
    boundary: int | None  # None: empty band, no recommendation
    block_aligned: bool

    @property
    def short_ceiling(self) -> int | None:
        if self.boundary is None:
            return None
        return self.boundary + self.delta - 1

    @property
    def long_floor(self) -> int | None:
        if self.short_ceiling is None:
            return None
        return self.short_ceiling + 1


def parse_ladder(text: str) -> list[LadderRow]:
    """Extract the (tokens, recompute_ms, pull_ms) rows from a Job log."""
    rows = []
    for line in text.splitlines():
        m = LADDER_ROW.match(line)
        if m:
            rows.append(
                LadderRow(
                    int(m["tokens"]),
                    float(m["recompute_ms"]),
                    float(m["pull_ms"]),
                )
            )
    return rows


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares intercept and slope of y = a + b * x."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("ladder needs at least two distinct token lengths")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def fit_pull_model(rows: list[LadderRow]) -> PullModel:
    """Fit pull_ms = floor + per_token * tokens over the ladder rows."""
    if len(rows) < 2:
        raise ValueError("ladder needs at least two rows to fit the pull model")
    floor_ms, slope_ms = _linear_fit(
        [float(r.tokens) for r in rows], [r.pull_ms for r in rows]
    )
    if slope_ms <= 0:
        raise ValueError("pull time does not grow with length; the ladder is not usable")
    return PullModel(floor_s=max(floor_ms, 0.0) / 1000.0, per_token_us=slope_ms * 1000.0)


def fit_throughput(rows: list[LadderRow]) -> float:
    """Peak prefill throughput (tokens/s) from the recompute column's slope."""
    if len(rows) < 2:
        raise ValueError("ladder needs at least two rows to fit the throughput")
    _, slope_ms = _linear_fit(
        [float(r.tokens) for r in rows], [r.recompute_ms for r in rows]
    )
    if slope_ms <= 0:
        raise ValueError("recompute time does not grow with length; the ladder is not usable")
    return 1000.0 / slope_ms


def derive(
    throughput: float,
    ttft_penalty_ms: float,
    block_size: int,
    max_context: int,
    pull: PullModel,
    delta: int = 1,
) -> Derivation:
    """Place the work-range boundary from the two anchors."""
    per_token_compute_us = 1e6 / throughput
    slo_tokens = throughput * ttft_penalty_ms / 1000.0
    if pull.per_token_us >= per_token_compute_us:
        crossover = None
    else:
        crossover = pull.floor_s * 1e6 / (per_token_compute_us - pull.per_token_us)

    boundary = None
    block_aligned = True
    if crossover is not None and crossover <= slo_tokens:
        aligned = math.ceil(crossover / block_size) * block_size
        aligned = max(aligned, block_size)
        if aligned <= slo_tokens:
            boundary = aligned
        else:
            # The band is narrower than one KV block: no block multiple fits.
            boundary = int(slo_tokens)
            block_aligned = boundary % block_size == 0
    return Derivation(
        throughput=throughput,
        per_token_compute_us=per_token_compute_us,
        pull=pull,
        ttft_penalty_ms=ttft_penalty_ms,
        block_size=block_size,
        max_context=max_context,
        delta=delta,
        crossover=crossover,
        slo_tokens=slo_tokens,
        boundary=boundary,
        block_aligned=block_aligned,
    )


def render(d: Derivation, pull_source: str, throughput_source: str) -> str:
    out = []
    out.append("== inputs")
    out.append(
        f"peak prefill throughput: {d.throughput:,.0f} tokens/s per rank "
        f"({throughput_source})"
    )
    out.append(
        f"pull model:              floor {d.pull.floor_s:.3f} s + "
        f"{d.pull.per_token_us:.1f} us/token ({pull_source})"
    )
    out.append(
        f"load gate budget:        {d.ttft_penalty_ms:.0f} ms (maxTTFTPenaltyMs)"
    )
    out.append(f"minCachedTokenDelta:     {d.delta}")
    out.append("")
    out.append("== derived anchors")
    out.append(
        f"per-token recompute:     {d.per_token_compute_us:.1f} us "
        f"(1e6 / throughput)"
    )
    if d.crossover is None:
        out.append(
            "pull crossover:          NONE, the per-token pull cost is at or "
            "above the per-token recompute cost; pulls never pay, do not split "
            "the pool"
        )
        return "\n".join(out)
    out.append(
        f"pull crossover:          {d.crossover:,.0f} tokens "
        "(below this, recompute beats pulling)"
    )
    out.append(
        f"SLO-equivalent work:     {d.slo_tokens:,.0f} tokens "
        f"({d.ttft_penalty_ms:.0f} ms at measured throughput)"
    )
    if d.boundary is None:
        out.append("")
        out.append("EMPTY BAND: the crossover exceeds the SLO-equivalent work.")
        out.append("Pulls cannot pay off within the declared TTFT budget.")
        out.append("Raise --ttft-penalty-ms or do not split the prefill pool.")
        return "\n".join(out)
    out.append("")
    out.append("== recommendation")
    out.append(
        f"boundary: {d.boundary:,d} tokens "
        f"(band [{d.crossover:,.0f}, {d.slo_tokens:,.0f}], "
        f"{'block-aligned' if d.block_aligned else 'NOT block-aligned'})"
    )
    if not d.block_aligned:
        out.append(
            f"WARNING: the band is narrower than one KV block ({d.block_size} "
            "tokens), so no block multiple fits under the budget; the boundary "
            "is the SLO-equivalent work instead. Raise --ttft-penalty-ms by a "
            "few hundred ms to get a block-aligned boundary."
        )
    if d.delta > 1:
        out.append(
            f"delta rule: the plugin routes a follow-up on newTokens + D - 1, so "
            f"the short ceiling is {d.boundary:,d} + {d.delta - 1:,d} = "
            f"{d.short_ceiling:,d} and the long range starts at {d.long_floor:,d}."
        )
    out.append("")
    out.append("== config to apply")
    out.append(
        f"""# short-work prefill class (pod template label)
llm-d.ai/prefill-work-range: "0-{d.short_ceiling}"
# long-work prefill class
llm-d.ai/prefill-work-range: "{d.long_floor}-{d.max_context}"
# both classes keep the full logical context range
llm-d.ai/context-length-range: "0-{d.max_context}"

# EPP plugin parameters (keep consistent with the labels above)
- type: prefix-cache-affinity-filter
  parameters:
    peakPrefillThroughput: {d.throughput:.0f}
    maxTTFTPenaltyMs: {d.ttft_penalty_ms:.0f}
- type: p2p-source-producer
  name: p2p-cache-source
  parameters:
    minCachedTokenDelta: {d.delta}
- type: context-length-aware
  parameters:
    label: llm-d.ai/prefill-work-range
    enableFiltering: false
    reusableTokensProducerName: p2p-cache-source"""
    )
    return "\n".join(out)


def _positive(kind):
    def convert(value: str):
        try:
            number = kind(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
        if not math.isfinite(number) or number <= 0:
            raise argparse.ArgumentTypeError(f"{value!r} must be finite and positive")
        return number

    return convert


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be finite and non-negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--peak-prefill-throughput",
        type=_positive(float),
        help="tokens/s of one rank, the PEAK_PREFILL_THROUGHPUT printed by "
        "calibrate.sh; fitted from the ladder's recompute column when omitted",
    )
    ap.add_argument(
        "--ttft-penalty-ms",
        type=_positive(float),
        required=True,
        help="tolerated added TTFT for the short class, the load gate's "
        "maxTTFTPenaltyMs",
    )
    ap.add_argument(
        "--block-size",
        type=_positive(int),
        default=64,
        help="the engines' --block-size in tokens (default 64, the P2P guides' value)",
    )
    ap.add_argument(
        "--max-context",
        type=_positive(int),
        required=True,
        help="the deployed --max-model-len; ceiling of the long work range",
    )
    ap.add_argument(
        "--min-cached-token-delta",
        type=_positive(int),
        default=1,
        help="the p2p-source-producer's minCachedTokenDelta (default 1)",
    )
    pull = ap.add_argument_group("pull model (one of)")
    pull.add_argument(
        "--pull-ladder",
        metavar="FILE",
        help="saved log of the calibrate-min-cached-token-delta.sh Job; the "
        "tokens / recompute_ms / pull_ms rows are fitted",
    )
    pull.add_argument(
        "--pull-floor-s",
        type=_non_negative_float,
        help="fixed cost of one P2P pull in seconds",
    )
    pull.add_argument(
        "--pull-per-token-us",
        type=_positive(float),
        help="marginal pull cost per token in microseconds",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    explicit = args.pull_floor_s is not None or args.pull_per_token_us is not None
    if args.pull_ladder and explicit:
        ap.error("give either --pull-ladder or both --pull-floor-s and --pull-per-token-us")
    if explicit and (args.pull_floor_s is None or args.pull_per_token_us is None):
        ap.error("--pull-floor-s and --pull-per-token-us go together")
    if not args.pull_ladder and not explicit:
        ap.error("a pull model is required: --pull-ladder FILE, or --pull-floor-s with --pull-per-token-us")

    rows: list[LadderRow] = []
    if args.pull_ladder:
        try:
            with open(args.pull_ladder, encoding="utf-8") as fh:
                rows = parse_ladder(fh.read())
        except OSError as exc:
            ap.error(f"cannot read --pull-ladder: {exc}")
        if len(rows) < 2:
            ap.error(
                f"--pull-ladder has {len(rows)} usable row(s); need at least two "
                "'tokens recompute_ms pull_ms' rows from the calibration Job"
            )
        try:
            pull = fit_pull_model(rows)
        except ValueError as exc:
            ap.error(str(exc))
        pull_source = f"fitted over {len(rows)} ladder rows"
    else:
        pull = PullModel(floor_s=args.pull_floor_s, per_token_us=args.pull_per_token_us)
        pull_source = "given"

    if args.peak_prefill_throughput is not None:
        throughput = args.peak_prefill_throughput
        throughput_source = "given"
    else:
        if not rows:
            ap.error("--peak-prefill-throughput is required without --pull-ladder")
        try:
            throughput = fit_throughput(rows)
        except ValueError as exc:
            ap.error(str(exc))
        throughput_source = "fitted from the ladder's recompute column"

    d = derive(
        throughput=throughput,
        ttft_penalty_ms=args.ttft_penalty_ms,
        block_size=args.block_size,
        max_context=args.max_context,
        pull=pull,
        delta=args.min_cached_token_delta,
    )
    print(render(d, pull_source, throughput_source))
    return 0 if d.boundary is not None else EXIT_NO_RECOMMENDATION


if __name__ == "__main__":
    sys.exit(main())

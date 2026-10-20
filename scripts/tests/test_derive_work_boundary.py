import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "guides" / "recipes" / "router" / "calibration" / "derive_work_boundary.py"

# GLM-5.2-FP8 on H200 reference inputs from the remaining-work routing study:
# 3585 tokens/s per rank, pull floor 1.25 s plus 134 us/token, 3500 ms gate.
GLM = ("--peak-prefill-throughput", "3585", "--ttft-penalty-ms", "3500",
       "--block-size", "64", "--max-context", "120000")
GLM_PULL = ("--pull-floor-s", "1.25", "--pull-per-token-us", "134")


def load_module():
    spec = importlib.util.spec_from_file_location("derive_work_boundary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def glm_ladder_text(floor_ms: float = 1250.0, per_token_ms: float = 0.134,
                    throughput: float = 3585.0) -> str:
    """A Job log in the calibrate-min-cached-token-delta.sh print format."""
    header = (
        "src=http://10.0.0.1:8200 dst=http://10.0.0.2:8200 pull_source=10.0.0.1:7777 "
        "model=zai-org/GLM-5.2-FP8 lengths=[2048, 4096, 8192, 16384, 32768] reps=3 dp_ranks=1"
    )
    lines = [
        header,
        "mesh warm: first pull 2100 ms (one-time session cost, discarded)",
        f"{'tokens':>8} {'recompute_ms':>13} {'pull_ms':>10} {'delta':>8} {'pulled_MB':>10}",
    ]
    for n in (2048, 4096, 8192, 16384, 32768):
        mr = n / throughput * 1000.0
        mp = floor_ms + per_token_ms * n
        d = (mp - mr) / mr * 100.0
        lines.append(f"{n:>8} {mr:>13.1f} {mp:>10.1f} {d:>+7.1f}% {n * 0.0056:>10.1f}")
    lines.append("MIN_CACHED_TOKEN_DELTA=16384")
    return "\n".join(lines) + "\n"


def test_glm_reference_reproduces_the_published_boundary() -> None:
    m = load_module()
    d = m.derive(3585.0, 3500.0, 64, 120000, m.PullModel(1.25, 134.0))

    assert 8600 < d.crossover < 8640
    assert d.boundary == 8640
    assert d.block_aligned
    assert d.short_ceiling == 8640
    assert d.long_floor == 8641
    assert d.slo_tokens == pytest.approx(12547.5)


def test_cli_prints_the_labels_and_epp_block() -> None:
    result = run_script(*GLM, *GLM_PULL)

    assert result.returncode == 0, result.stderr
    assert "boundary: 8,640 tokens" in result.stdout
    assert 'llm-d.ai/prefill-work-range: "0-8640"' in result.stdout
    assert 'llm-d.ai/prefill-work-range: "8641-120000"' in result.stdout
    assert "peakPrefillThroughput: 3585" in result.stdout
    assert "maxTTFTPenaltyMs: 3500" in result.stdout
    assert "reusableTokensProducerName: p2p-cache-source" in result.stdout


def test_delta_shifts_the_short_ceiling_by_delta_minus_one() -> None:
    m = load_module()
    d = m.derive(3585.0, 3500.0, 64, 120000, m.PullModel(1.25, 134.0), delta=2048)

    assert d.boundary == 8640
    assert d.short_ceiling == 8640 + 2047
    assert d.long_floor == 8640 + 2048

    result = run_script(*GLM, *GLM_PULL, "--min-cached-token-delta", "2048")
    assert 'llm-d.ai/prefill-work-range: "0-10687"' in result.stdout
    assert 'llm-d.ai/prefill-work-range: "10688-120000"' in result.stdout
    assert "newTokens + D - 1" in result.stdout


def test_band_narrower_than_a_block_is_reported_not_silently_misaligned() -> None:
    # 2408 ms puts the SLO work at 8632.7 tokens: above the 8624 crossover but
    # below the next block multiple (8640), so no aligned boundary fits.
    m = load_module()
    d = m.derive(3585.0, 2408.0, 64, 120000, m.PullModel(1.25, 134.0))

    assert d.crossover < d.slo_tokens < 8640
    assert d.boundary == 8632
    assert not d.block_aligned

    result = run_script(*GLM[:2], "--ttft-penalty-ms", "2408", *GLM[4:], *GLM_PULL)
    assert result.returncode == 0
    assert "NOT block-aligned" in result.stdout
    assert "WARNING: the band is narrower than one KV block" in result.stdout


def test_empty_band_gives_no_recommendation() -> None:
    result = run_script(*GLM[:2], "--ttft-penalty-ms", "2000", *GLM[4:], *GLM_PULL)

    assert result.returncode == 1
    assert "EMPTY BAND" in result.stdout
    assert "== recommendation" not in result.stdout
    assert "Traceback" not in result.stderr


def test_pull_slower_than_recompute_never_pays() -> None:
    result = run_script(*GLM, "--pull-floor-s", "1.25", "--pull-per-token-us", "300")

    assert result.returncode == 1
    assert "pull crossover:          NONE" in result.stdout
    assert "do not split" in result.stdout


def test_ladder_fit_recovers_the_pull_model_and_throughput(tmp_path: Path) -> None:
    m = load_module()
    rows = m.parse_ladder(glm_ladder_text())
    assert [r.tokens for r in rows] == [2048, 4096, 8192, 16384, 32768]

    pull = m.fit_pull_model(rows)
    assert pull.floor_s == pytest.approx(1.25, abs=1e-3)
    assert pull.per_token_us == pytest.approx(134.0, abs=0.05)
    assert m.fit_throughput(rows) == pytest.approx(3585.0, rel=1e-3)

    log = tmp_path / "ladder.log"
    log.write_text(glm_ladder_text())
    result = run_script("--ttft-penalty-ms", "3500", "--block-size", "64",
                        "--max-context", "120000", "--pull-ladder", str(log))
    assert result.returncode == 0, result.stderr
    assert "fitted over 5 ladder rows" in result.stdout
    assert "fitted from the ladder's recompute column" in result.stdout
    assert "boundary: 8,640 tokens" in result.stdout


def test_ladder_rows_are_parsed_only_from_the_table() -> None:
    m = load_module()
    text = (
        "RUN DISCARDED: the no-pull control moved bytes, so the\n"
        "    2048         571.2     1524.4  +166.9%       11.5\n"
        "  tokens  recompute_ms    pull_ms    delta  pulled_MB\n"
        "MIN_CACHED_TOKEN_DELTA=2048\n"
    )
    rows = m.parse_ladder(text)
    assert len(rows) == 1
    assert rows[0] == m.LadderRow(2048, 571.2, 1524.4)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((*GLM, "--pull-floor-s", "1.25"), "go together"),
        ((*GLM,), "a pull model is required"),
        ((*GLM[:2], "--ttft-penalty-ms", "0", *GLM[4:], *GLM_PULL), "finite and positive"),
        ((*GLM[:2], "--ttft-penalty-ms", "inf", *GLM[4:], *GLM_PULL), "finite and positive"),
        ((*GLM, "--pull-floor-s", "-1", "--pull-per-token-us", "134"), "finite and non-negative"),
        (("--ttft-penalty-ms", "3500", "--block-size", "64", "--max-context", "120000",
          "--pull-ladder", "/nonexistent/ladder.log"), "cannot read --pull-ladder"),
    ],
)
def test_invalid_parameters_are_reported_without_traceback(
    args: tuple[str, ...], message: str
) -> None:
    result = run_script(*args)

    assert result.returncode != 0
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_ladder_with_one_length_is_rejected(tmp_path: Path) -> None:
    log = tmp_path / "ladder.log"
    log.write_text("    8192        2285.4     2347.7    +2.7%       45.9\n")

    result = run_script("--ttft-penalty-ms", "3500", "--max-context", "120000",
                        "--pull-ladder", str(log))
    assert result.returncode != 0
    assert "need at least two" in result.stderr
    assert "Traceback" not in result.stderr

# Calibrating `peakPrefillThroughput`

Router configs that use the `prefix-cache-affinity-filter` plugin set
`peakPrefillThroughput` on it. The filter uses this value to estimate per-endpoint
time-to-first-token from in-flight load, which drives prefix-cache-aware routing.

The value is **hardware- and model-specific** — the plugin default (`15928`) is
calibrated for Qwen 32B on H100 80 GB (TP=2). If you deploy a different model or accelerator, measure your
own with this tool and set it on the `prefix-cache-affinity-filter` plugin in your
guide's router values file. (The Qwen3-Coder-480B guide ships `16444`, measured for
Qwen3-Coder-480B-FP8 on TPU v7x.)

See the [**configuration matrix**](./configuration-matrix.md) for the reference values by
(model, accelerator) shipped under `guides/`, and which combinations still need a calibration run.

## What it measures

`calibrate.sh` runs a short Kubernetes Job ([`calibration-peak-throughput.yaml`](calibration-peak-throughput.yaml))
that sends warmup + measurement requests of exactly `CHUNK_SIZE` random token IDs (so the
prefix cache misses every time and we measure true prefill), records TTFT, and computes:

```text
peakPrefillThroughput = CHUNK_SIZE / median(TTFT)   # tokens/sec
```

It **only measures and prints** the value — it does not modify any config.

## Prerequisites

- The stack is deployed and serving (router + model server), reachable from the Job's network.
- `kubectl` and `envsubst` on your `PATH`.

## Usage

```bash
GUIDE_NAME=qwen3-coder-480b \
NAMESPACE=llm-d-qwen3-coder-480b \
MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
CHUNK_SIZE=8192 \
./calibrate.sh
```

| env var | meaning | default |
| --- | --- | --- |
| `GUIDE_NAME` | release/guide name (used for the `<name>-epp` service) | `optimized-baseline` |
| `NAMESPACE` | namespace the stack runs in | `default` |
| `MODEL_NAME` | model vLLM is serving | `Qwen/Qwen3-32B` |
| `CHUNK_SIZE` | request size; **must match vLLM `--max-num-batched-tokens`** | `8192` |
| `T_MAX_SECONDS` | TTFT SLO tolerance (informational `TAU` line only) | `18` |
| `VLLM_ENDPOINT` | `http://host:port`; auto-discovered from the EPP service if unset | — |
| `NUM_WARMUP` / `NUM_MEASUREMENTS` | request counts | `5` / `20` |

## Applying the value

Set the measured number on the `prefix-cache-affinity-filter` plugin in your guide's
router values file:

```yaml
- type: prefix-cache-affinity-filter
  parameters:
    peakPrefillThroughput: <measured value>
```

Then re-apply the router release (`helm upgrade ... -f <your-guide>.values.yaml`) and
restart the EPP:

```bash
kubectl rollout restart -n ${NAMESPACE} deployment/${GUIDE_NAME}-epp
```

## Files

| file | purpose |
| --- | --- |
| `calibrate.sh` | orchestration: runs the Job, extracts and prints the value |
| `calibration-peak-throughput.yaml` | the measurement Job + its Python script (ConfigMap) |

## Calibrating `minCachedTokenDelta`

Router configs that use the `p2p-source-producer` (the
[p2p-kv-cache-sharing guide](../../../p2p-kv-cache-sharing/README.md)) set
`minCachedTokenDelta` on it: a pull is requested only when a peer holds at
least that many more cached prefix tokens than the scheduled pod. Below the
pull-versus-recompute crossover a pull costs more than recomputing, so the
right value is the crossover — and the crossover is **model-, hardware- and
transport-specific** (measured on gpt-oss-120b/H200: below 2K with `rdma/ib`
on the pods; calibrate separately on the TCP fallback).

[`calibrate-min-cached-token-delta.sh`](calibrate-min-cached-token-delta.sh)
runs a Job ([`calibration-min-cached-token-delta.yaml`](calibration-min-cached-token-delta.yaml))
that measures it against two live model-server pods: per length and
repetition it seeds a fresh random token-ID prompt on the source pod, then
times the consumer pod serving that prompt with and without
`kv_transfer_params.remote_kv_source` (independent prompts for each leg,
because the consumer caches whatever it just served). The mesh is warmed
first so the one-time session-establishment cost is excluded. It prints the
measured ladder and the recommendation:

```text
MIN_CACHED_TOKEN_DELTA=<smallest tested length where the pull won>
```

```bash
NAMESPACE=llm-d-p2p \
POD_SELECTOR=llm-d.ai/guide=p2p-kv-cache-sharing \
MODEL_NAME=openai/gpt-oss-120b \
./calibrate-min-cached-token-delta.sh
```

Unlike the peak-throughput Job this cannot go through the router — the pull
is driven by injecting `kv_transfer_params` directly at two engine
endpoints — so the script talks to pod IPs (`ENGINE_PORT`, default `8200`).
Prerequisites: the OffloadingConnector P2P tier on the pods,
`PYTHONHASHSEED` pinned fleet-wide, and the same transport you will deploy
on. Lengths must be multiples of the vLLM block size.

## Deriving the prefill work-range boundary

Remaining-work routing (`context-length-aware` with `reusableTokensProducerName`,
see the [plugin README](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/scheduling/scorer/contextlengthaware/README.md))
splits the prefill pool into a short-work and a long-work class by the
`llm-d.ai/prefill-work-range` label and routes a cached follow-up by the
prefill work left after its pullable prefix. The boundary between the two
classes follows from the two numbers above, so it needs no further measurement:
a pull pays only above the pull-versus-recompute crossover, and a request is
long work once one rank needs longer than the load gate's `maxTTFTPenaltyMs`
to prefill it.

[`derive_work_boundary.py`](derive_work_boundary.py) takes `peakPrefillThroughput`
(from `calibrate.sh`) and either the saved output of
`calibrate-min-cached-token-delta.sh` (it fits the pull floor and per-token
cost from the `tokens recompute_ms pull_ms` rows) or an explicit pull model,
and prints the anchors, the block-aligned boundary, and the labels and EPP
parameters to paste:

```bash
guides/recipes/router/calibration/derive_work_boundary.py \
  --peak-prefill-throughput 3585 --ttft-penalty-ms 3500 \
  --block-size 64 --max-context 120000 \
  --pull-ladder calibration-min-cached-token-delta.log
```

On the GLM-5.2-FP8 on H200 reference (3585 tokens/s per rank, pull floor
1.25 s plus 134 us per token, 3500 ms budget) it derives an 8,640-token
boundary. An empty band (crossover above the SLO-equivalent work) means pulls
cannot pay within the budget: raise `--ttft-penalty-ms` or do not split the
pool. With `minCachedTokenDelta` D above 1 the plugin routes a follow-up on
`newTokens + D - 1`, so the printed short ceiling is the boundary plus D - 1.
It derives only; it does not touch any config.

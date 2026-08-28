# GLM-5.3-Flash on 2× NVIDIA DGX Spark

Serve **GLM-5.3-Flash (NVFP4)** at 262 144 context across two DGX Sparks
(GB10, `sm_121`) with vLLM, fp8 KV cache, and DFlash2 speculative decoding.

Stock vLLM **cannot** run this model on GB10. This repo is the exact patch
stack, build chain, and launch procedure that makes it work, recovered from a
running deployment rather than written from memory.

---

## Why this needs patching at all

GLM-5.3-Flash uses **NoPE MLA** (`qk_rope_head_dim=0`). That is architecturally
different from DeepSeek's MLA (`pe_dim=64`), which is what vLLM's stock SM120
sparse-attention kernel (`FLASHINFER_MLA_SPARSE_SM120`) was written for. On
GB10 that kernel is the *only* stock sparse-MLA backend for compute capability
12, and it hard-crashes during warmup:

```
concat_and_cache_mla: pe_dim must be 64 for fp8_ds_mla
```

This was confirmed by reading the checkpoint's actual tensor shapes before
trusting any third-party claim: `kv_a_proj_with_mqa.weight` is `[512, 4096]` —
exactly `kv_lora_rank`, with no added rope dimension.

**The fix, in one sentence:** don't patch the broken SM120 kernel — instead
extend vLLM's **SM90 (Hopper) NoPE-MLA backend**'s capability gate to also
accept SM120/121, because that backend was never written with DeepSeek's
`pe_dim=64` assumption and handles `qk_rope_head_dim=0` correctly by
construction.

Every patch in this repo is guarded: each one asserts the exact source text it
expects and `raise SystemExit`s if the base image doesn't match, so a version
drift fails the build loudly instead of silently corrupting the image.

---

## Requirements

| | |
|---|---|
| Hardware | 2× DGX Spark (GB10, aarch64, `sm_121`), 128 GB unified memory each |
| Interconnect | The direct Spark-to-Spark 200G link (ConnectX). **Not** your LAN. |
| Disk | ~185 GB free on **local** NVMe on **both** nodes |
| Base image | `vllm/vllm-openai:glm53-flash-arm64-cu130` |
| Docker | With `--gpus all` and `/dev/infiniband` passthrough |

> **Put the weights on local disk.** Loading from local EXT4 takes ~3 min/rank.
> The same load over a CIFS/SMB network mount took 8–10 min/rank and made
> debugging miserable.

---

## 1. Fetch the models

```bash
# Target model (~182 GB) — NVFP4 quant of zai-org/GLM-5.3-Flash
hf download LibertAIDAI/GLM-5.3-Flash-NVFP4 \
    --local-dir /var/tmp/glm-5.3-flash-nvfp4

# DFlash2 draft model (~2.2 GB)
hf download incoai/GLM-5.3-Flash-DFlash2 \
    --local-dir /var/tmp/glm-5.3-flash-dflash2
```

Both must exist at those paths on **both** nodes.

> **Licensing:** the target model is MIT. The DFlash2 draft model is
> **CC-BY-NC-ND-4.0 — non-commercial, no derivatives.** Check it before using
> this in production. Neither model is redistributed here.

---

## 2. Build the image chain

Each layer is a separate, individually verifiable step. Build them in order —
every Dockerfile's `FROM` is the previous tag.

```bash
cd docker
./../scripts/build-all.sh
```

Or manually:

```bash
docker build -f Dockerfile.sm121-v1 -t glm53:sm121-v1 .
docker build -f Dockerfile.sm121-v2 -t glm53:sm121-v2 .
# ... through v8 ...
docker build -f Dockerfile.vllm-dflash2 -t glm53:vllm-dflash2-v1 .
```

| Layer | What it does | Why |
|---|---|---|
| **v1** | Extend `FLASHINFER_MLA_SPARSE_SM90`'s capability gate from `{9}` to `{9,12}`; add it to the capability-12 candidate list in `platforms/cuda.py`; switch FA3→FA2 off-Hopper; scope the FlashInfer ≥0.6.18 gate to only the fp8-KV path | **The core fix.** |
| **v2** | Env-gated NaN-localizing forward hooks (inert unless `GLM53_NAN_DEBUG=1`) | Debugging aid; keep it, it costs nothing |
| **v3** | FlashInfer → `0.6.18.dev20260819` nightly; uninstall stale `flashinfer-jit-cache` | 0.6.17's FA2 MLA scheduler returns **NaN for 64–256-row batches** on SM121 (bisected: 1–6 and 512+ rows are clean) |
| **v4** | Re-pin `nvidia-nccl-cu13==2.30.7` | The nightly silently downgrades it to 2.29.7, which fails `ncclCommInitRank` on the Spark IB fabric |
| **v5** | Re-pin `nvidia-cutlass-dsl==4.6.2` | The nightly leaves a mixed 4.7.0/4.6.2 install that ICEs CuTeDSL warmup |
| **v6** | Gate PDL off SM12x (`is_arch_support_pdl`: `major >= 9` → `major in (9,10)`) | Programmatic Dependent Launch races on KDA recurrent-state kernels on GB10 — boots NaN or not depending on launch timing |
| **v7** | Indexer hardening: kpool top-k buffer `torch.empty` → `torch.full(-1)`, plus a Triton bounds clamp | Uninitialized entries fed bogus token indices into MLA gathers → NaN lottery. **This is the validated bf16-KV baseline.** |
| **v8** | fp8 KV for the SM90 NoPE path: cap `EFF_CTA_TILE_KV` at 32 instead of forcing it, and accept major 12 in FlashInfer's `_core.py` gate | GB10's ~101 KB smem can't fit Hopper's 228 KB-smem tile assumption |
| **dflash2** | Port of upstream vLLM PR #52816 (DFlash2 drafter) onto v8, plus GLM-5.3 glue wiring `Glm5NextModel` to `SupportsEagle3` for aux hidden states, plus a KV-cache-group fix for the drafter's sliding-window layers | Speculative decoding. Replaces the earlier MTP-4 config. |

---

## 3. Launch

**Worker first (rank 1), then head (rank 0).** The worker runs `--headless`;
the head serves the API.

```bash
# On the worker node
./scripts/launch-glm53-vllm-dflash2-tp2.sh 1

# Then on the head node
./scripts/launch-glm53-vllm-dflash2-tp2.sh 0
```

Edit `HEAD_IP` in the script to your head node's **interconnect** address.
Cold start to serving is **~18 minutes** (120 shards at ~5.9 s each, then draft
load and FlashInfer/CuTeDSL warmup). The script does its own `docker rm -f`.

> **Keep the script identical on both nodes.** They are two separate files.
> A change applied to only one is half a change, and the mismatch is silent.

### Site-specific values you must change

The launch script is committed exactly as it runs in production, so these are
this cluster's values, not yours:

| Variable | Here | What it is |
|---|---|---|
| `HEAD_IP` / `HOST_IP` | `10.0.10.1` / `10.0.10.2` | Interconnect addresses of head (rank 0) and worker (rank 1) |
| `NCCL_IB_HCA` | `rocep1s0f1` | Your RoCE/IB HCA (`ibv_devinfo`) |
| `NCCL_SOCKET_IFNAME` etc. | `enp1s0f1np1` | The interconnect NIC (`ip -br link`) |
| `MODEL_HOST_PATH` / `DRAFT_HOST_PATH` | `/var/tmp/...` | Where you downloaded the weights |

`GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `MN_IF_NAME` must all match
`NCCL_SOCKET_IFNAME`.

### Flags that are load-bearing

- `--block-size 2304` — must be a multiple of both `index_kpool(4)·64 = 256`
  and the MLA 128-alignment (2304 = 256×9 = 128×18). Smaller values that seem
  reasonable (including 128) don't satisfy this backend's pool-page tiling and
  hit a DeepGEMM assert.
- `--gpu-memory-utilization 0.85` — headroom for the draft model on unified memory.
- `--enforce-eager` — CUDA graph capture is not viable on this path.
- `--reasoning-parser glm45` / `--tool-call-parser glm47` — see below.
- **Do not** copy `--kv-cache-memory=<bytes>` from any external recipe. It is
  tuned to one box's exact memory state at boot. Let it auto-compute; if vLLM
  logs a suggested value, use *that*.

---

## 4. Verify

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash",
       "messages":[{"role":"user","content":"What is 23*17? Think briefly."}],
       "max_tokens":400,"temperature":0}' | jq '.choices[0].message'
```

A correct response has reasoning **separated**, not inlined:

```json
{
  "role": "assistant",
  "reasoning": "23 * 17 = 23 * 20 - 23 * 3 = 460 - 69 = 391.",
  "content": "**391**\n\n(23 × 17 = 23 × 20 − 23 × 3 = 460 − 69 = 391)"
}
```

If `reasoning` is `null` and the chain-of-thought is glued to the front of
`content`, read [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#reasoning-tokens-end-up-in-content) —
there is a specific flag that causes exactly this.

---

## Attribution

- The patch approach originated in a community repo
  (`tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark`), verified file-by-file
  before use and rebuilt locally rather than trusting its prebuilt image refs.
- `docker/qwen3_dflash2.py` and `docker/dflash2/` are from **vLLM PR #52816**
  (merged 2026-08-21, `b389ac294`), Apache-2.0, © vLLM contributors. SPDX
  headers are intact. `dflash2/speculator.py` carries one adaptation: a local
  `gumbel_noised_argmax` copy, because this image's `gumbel.py` predates the
  PR's refactor.
- DFlash2 is by [z-lab](https://github.com/z-lab/dflash) / inco.ai.

The patch scripts and Dockerfiles in this repo are MIT (see LICENSE). The vLLM
code they patch and vendor remains Apache-2.0.

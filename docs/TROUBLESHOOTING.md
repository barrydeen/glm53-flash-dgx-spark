# Troubleshooting

Failure modes actually hit on this hardware, with the diagnosis that resolved
each one.

---

## Reasoning tokens end up in `content`

**Symptom.** Apps show the model's chain-of-thought concatenated to the front
of the answer. `message.reasoning` is `null`. No `<think>` tags are visible
anywhere — the CoT just runs straight into the response.

**Cause.** A launch flag that looks like it disables thinking, but actually
disables *parsing* while leaving thinking on:

```
--default-chat-template-kwargs '{"enable_thinking": false}'
```

The two halves of the stack disagree about who owns that key:

- **The chat template ignores it.** GLM-5.3's `chat_template.jinja` contains
  **zero** occurrences of `enable_thinking` — it gates thinking on
  `clear_thinking` and the `/nothink` token (154851). Its last line emits
  `<|assistant|><think>` unconditionally, so the model reasons on every request.
- **The parser obeys it.** `--reasoning-parser glm45` resolves to
  `Glm47MoeParser` (`vllm/parser/glm47_moe.py`). Its `__init__` reads
  `enable_thinking` out of the chat-template kwargs and sets
  `thinking_enabled = False`, which makes `extract_reasoning()` return
  `(None, model_output)` and builds the state machine with
  `initial_state=CONTENT` and no `<think>` terminals at all.

Net effect: thinking stays **on**, parsing turns **off**.

**Fix.** Remove the flag entirely. With both `thinking` and `enable_thinking`
unset, `thinking_enabled` defaults to `True`.

Removing it is correct rather than setting it to `true`, and the reason
matters: because the template injects `<think>` into the *prompt*, the model
only ever emits the **closing** `</think>`, never an opening tag. The parser
handles this by starting in `REASONING` state — which it only does when
thinking is enabled. A parser that required a `<think>` in the output would
fail here no matter what.

**Two follow-ons:**

1. This vLLM emits the field as **`reasoning`**, not `reasoning_content`. The
   output schema is `ChatMessage.reasoning`; an incoming `reasoning_content` is
   renamed *to* `reasoning`, but the old name is never emitted. Clients
   hardcoded to `reasoning_content` will show an empty reasoning pane even when
   the server is correct.
2. **Any app-side "disable thinking" toggle re-triggers this per-request**,
   since it sends the same kwarg. Grep your front-ends for `enable_thinking`
   before calling this closed.

**Verify all three paths** — they are separate code paths in the parser:

```bash
# non-streaming: .reasoning populated, .content clean
# streaming:     delta.reasoning deltas arrive separately
# tool calling:  finish_reason=tool_calls, valid JSON args, reasoning still split
```

Tool calling is worth re-testing specifically, because enabling thinking
changes the parser's initial state and activates a different
`REASONING → TOOL_START` transition.

---

## Box hard-freezes ~20 min into model load

**Cause.** GB10 unified memory: page cache is allowed to grow into memory the
GPU allocator needs, and the box locks up silently.

**Fix.** Before every launch, on **both** nodes:

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

In practice `docker rm -f` of the previous container already releases nearly
everything — measured 2–3 GB buff/cache and 116 GB available immediately after
teardown, which is the same state the drop produces. Check `free -g` before
deciding whether you need to chase a sudo password.

---

## `NV_ERR_NO_MEMORY` during startup

Not enough room for the draft head alongside the KV cache. fp8 KV (v8) roughly
halves the cache footprint and frees the ~2–5 GB the drafter needs. If it still
OOMs, lower `--gpu-memory-utilization` before touching anything else, and let
`--kv-cache-memory` auto-compute.

---

## NaN output

Three distinct causes were found, in this order. If you are on the full v8+
chain all three are already fixed; this is for diagnosing a *partial* build.

1. **FlashInfer 0.6.17's FA2 MLA scheduler** returns NaN for 64–256-row
   batches on SM121. 1–6 and 512+ rows are clean, which makes it look
   load-dependent. → v3.
2. **PDL races** on KDA recurrent-state kernels (`causal_conv1d`,
   `gather_initial_states`, `scatter_states`). Timing-dependent: full-device-sync
   debug hooks reliably *mask* it, which is a strong tell. → v6.
3. **Uninitialized kpool top-k entries.** The top-k kernels only guarantee the
   first `min(k, valid)` entries; rows with fewer valid pools carried garbage
   pool ids → bogus token indices → MLA gathers uninitialized KV. → v7.

To localize a *new* NaN, build with v2 in the chain and set `GLM53_NAN_DEBUG=1`;
it logs the first module whose output goes non-finite.

---

## `pe_dim must be 64 for fp8_ds_mla`

You are on the stock SM120 sparse-MLA kernel, which assumes DeepSeek's MLA
shape. This is the failure the whole v1 patch exists to avoid — see the README.

---

## `ncclCommInitRank` internal error

`nvidia-nccl-cu13` got downgraded to 2.29.7, which doesn't work on the Spark IB
fabric. Re-pin 2.30.7 (v4). Check with `pip show nvidia-nccl-cu13`.

Also confirm you are using the **interconnect** addresses (the direct 200G
link), not the LAN addresses, for `--master-addr` and `VLLM_HOST_IP`.

---

## CuTeDSL warmup ICE (`cute-to-nvvm`)

Mixed `nvidia-cutlass-dsl` 4.7.0/4.6.2 install left by the FlashInfer nightly.
Re-pin 4.6.2 (v5).

---

## DeepGEMM assert on `--block-size`

`--block-size` must be a multiple of both `index_kpool(4)·64 = 256` and the MLA
128-alignment. Use **2304** (= 256×9 = 128×18). Values derived for the
*different*, abandoned SM120-sparse path — including 128 — do not satisfy this
backend's pool-page tiling.

---

## Multimodal / ViT encoder dies at startup

`cudaErrorUnsupportedPtxVersion` — the `vllm_flash_attn` build has no `sm_121`
cubin, so the encoder's FA2 path fails. Use
`--mm-encoder-attn-backend TORCH_SDPA` (correct, slower). The LM path picks
FlashInfer and is unaffected.

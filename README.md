# ibm-hf-convert

GitHub Actions pipeline that turns a Hugging Face model into **big-endian** GGUF files (for s390x / IBM Z)
and uploads them to `<namespace>/<Model>-BE-GGUF` on Hugging Face. It runs on Blacksmith ARM runners.

The model name is title-cased everywhere it appears: the HF repo, the GGUF file names and the `models/` folder.
So `ibm-granite/granite-4.2-3b` becomes `Granite-4.2-3B-BE-GGUF`, `Granite-4.2-3B-BE-Q4_K_M.gguf` and
`models/Granite-4.2-3B/`.
- Size labels are upper-cased: `3b` → `3B`, `8x7b` → `8x7B`.
- `it` and `vl` become `IT` and `VL`.
- Parts that already have capitals, and version parts like `v0.3`, are left alone.

If a repo with the same name in different case already exists, it is reused, and the run notes that you can
rename it on HF.

## Usage

Go to **Actions → Convert to big-endian GGUF → Run workflow**, then enter a model tag:

| Input | Default | Notes |
|---|---|---|
| `model` | — | `Qwen/Qwen3-8B` (safetensors) or `unsloth/Qwen3-8B-GGUF` (GGUF). Add `:Q4_K_M` to a GGUF repo to convert only that quant. |
| `quants` | `NATIVE,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q2_K,Q3_K_S,Q3_K_M,Q3_K_L,Q4_K_S,Q4_K_M,Q5_K_S,Q5_K_M,Q6_K,IQ4_NL,IQ4_XS,MXFP4_MOE` | Safetensors models only. `NATIVE` is the source's own float type (see below). |
| `hf_namespace` | `HF_NAMESPACE` variable, else the token owner | User or org to upload to. |
| `runner` | `auto` | Sets the runner size. `auto` picks one as described in [Runner sizing](#runner-sizing). |
| `target_minutes` | `15` | Used by `auto`: the time budget for quantizing. Raise it to use smaller, cheaper runners. |
| `private` | `false` | Applies only when the output repo is created. |
| `llama_cpp_ref` | `latest` | `latest` or a release tag like `b11718` uses the prebuilt ARM tools from that release. A branch or commit is built from source, which takes about 2.5 min longer. |

### Float outputs: `NATIVE`

`NATIVE` in the quant list stands for the float type that holds most of the source's weights: BF16 for a
BF16-trained model, F16 for an F16 one, F32 for an F32 one. FP8 or integer sources get BF16. For a BF16 model,
the other two formats add nothing:
- **F32** stores the same values at twice the size, and decodes more slowly.
- **F16** has less exponent range than BF16, so the conversion can lose values.

On a z17 (2 vCPU, Granite-4.0-H-1B) BF16 was also faster than F16:

| Format | Prompt (pp128) | Generation (tg32) |
|---|---|---|
| BF16 | 15.0 tok/s | 1.95 tok/s |
| F16 | 8.9 tok/s | 1.36 tok/s |

F32 couldn't be measured fairly: the 5.4 GiB file is larger than that VM's 4 GB of RAM. `F16` and `F32` can
still be requested explicitly.

### What happens

1. **Plan:** a job on GitHub's free `ubuntu-slim` runner reads the repo's file list from the HF API.
   - If the repo has `*.safetensors`, it runs the safetensors pipeline. If it has only `*.gguf`, it runs the GGUF pipeline.
   - It checks the requested quant types. It drops `MXFP4_MOE` for dense models, where the output would be a copy of `Q8_0`.
   - It estimates disk need and picks the smallest runner with enough disk.
2. **Safetensors pipeline:**
   1. Download the model. If the repo ships Mistral's native format (`params.json`, `consolidated*.safetensors`,
      `tekken.json`), download that instead of the HF-format weights and convert with `--mistral-format`.
      Newer Mistral models need this, because their HF tokenizer only loads in transformers 5.
   2. If F32/F16 were requested and any source tensor is not BF16, make them directly from the safetensors.
   3. Make a little-endian BF16 intermediate. For vision or audio models (`vision_config`/`audio_config` in
      `config.json`), also export an F16 projector, `mmproj-<Model>-BE-F16.gguf`. Then delete the safetensors.
   4. Run `llama-quantize` once per remaining type. If every source tensor is BF16, this includes F32 and F16:
      the BF16 intermediate holds the source weights exactly, and `llama-quantize` is much faster than another
      `convert_hf_to_gguf.py` pass, which is mostly single-threaded and takes about 39 s per billion parameters.
      Outputs over 48 GB are split with `llama-gguf-split`.
   5. Byte-swap each output to big-endian, upload it and delete it locally. This runs on background threads
      while the next output is converted or quantized: 1 thread below 16 vCPU, 2 on 16 vCPU, 3 on 32 vCPU.
      A new output starts only when the disk has room for it.
   6. The BF16 output is byte-swapped and uploaded from a copy right after the intermediate is made, if the disk
      has room for that copy next to a full queue of quants. Otherwise it goes last, because the quant steps
      need the intermediate in little-endian.
3. **GGUF pipeline:** handles every `*.gguf` in the repo, including mmproj files and split shards, one at a time:
   1. Download the file.
   2. Byte-swap it to big-endian.
   3. Upload it as `<Model>-BE-<QUANT>.gguf` and delete it locally. The next file downloads while this one uploads.

   Files with tensor types the endian script can't handle (such as IQ2/IQ3 quants) are skipped with a warning.

Byte-swapping uses llama.cpp's `gguf-py/gguf/scripts/gguf_convert_endian.py`. Every output is checked to be
big-endian before upload. Files are named `<Model>-BE-<QUANT>.gguf` (for example `Bonsai-8B-BE-Q1_0.gguf`).
`Q3_K`, `Q4_K` and `Q5_K` are llama.cpp aliases for the `_M` types, so their files are named `Q3_K_M`, `Q4_K_M`
and `Q5_K_M`. HF's `-hf repo:TAG` lookup only accepts the full names.

Every GGUF also gets the uploading HF account (for example `taronaeo`) added to its `general.tags` metadata:
- **Safetensors pipeline:** passed to `convert_hf_to_gguf.py --metadata`, next to the tags the converter takes
  from the model card. `llama-quantize` then carries it into every quant.
- **GGUF pipeline:** the source file's metadata is rewritten before byte-swapping, which briefly needs twice the
  file's size on disk. Shards after the first have no `general.*` metadata and aren't rewritten.
Split outputs go under `<QUANT>/`.

At the end of every run, the repo's `README.md` is regenerated in one commit:
- **Front matter:** the original card's front matter, with `base_model` pointing at the source and these
  tags added: `gguf`, `mainframe`, `s390x`, `z15`, `z16`, `z17`, `big-endian`.
- **Header:** a link to llama.cpp's [IBM Z & LinuxONE guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build-s390x.md),
  the model creator and original model, and a `llama-server -hf <repo>:Q4_K_M` example. The example uses the
  quant closest to 4 bits when there is no Q4_K_M.
- **Provided Files:** a table listing every GGUF in the repo, including files from earlier runs.
- **Original card:** the source model's card, unchanged, after a `---` divider.

The original's LICENSE/NOTICE files, images (plus any file the card links to by relative path) and
`.eval_results/` are copied too, so card images still render.
Any manual edits to the README are overwritten on the next run.

A final **Save model card** job on `ubuntu-slim` also commits the same card to this repo as
`models/<Model>/README.md`, for example `models/Bonsai-8B/README.md`. The copied images and license files go next to it, so
the GitHub copy renders the same way. It runs even if some quants failed, since the card
still lists everything that was uploaded. It needs push access to the branch the workflow ran from, so if
that branch is protected, allow `github-actions[bot]` to push to it.
If a folder with the same name in different case already exists, it is replaced.

### When a run fails

If the conversion job fails or is cancelled, a **Delete HF repo after failure** job removes the output repo, but
only if the planner saw that it didn't exist before this run. A repo from earlier runs is never deleted, so its
existing uploads stay. Its card is still refreshed to list what is there. The **Save model card** job is skipped
for deleted repos, so nothing lands in `models/`. Models that llama.cpp can't convert (for example custom-code
architectures like `GraniteVisionEmb`) fail this way and leave nothing behind.

### Runner sizing

`auto` picks the smallest runner that meets two conditions:

1. **Disk:** it has room for everything that exists at the same time: the safetensors, the BF16 intermediate and
   the current outputs. Not enough disk is the one thing that can make a run fail.
2. **CPU time:** its estimated quantize time fits within `target_minutes`. Quantizing is CPU-bound: a
   granite-4.2-3b run on `4vcpu` sat at 99% CPU and about 15% memory. Runners are billed per vCPU-minute, so
   for that part more vCPUs cost about the same and finish sooner. The estimate uses each quant type's cost
   measured on that run (`QUANT_CPU_COST` in `scripts/common.py`). It also accounts for `llama-quantize` not
   scaling linearly with threads (`THREAD_EFFICIENCY`): about 100% up to 8 vCPU, 75% at 16, and 55% at 32,
   measured on a granite-4.2-30b run. If no runner meets the target, the largest runner with enough disk is used.

RAM is not a constraint. `convert_hf_to_gguf.py` loads tensors lazily, and `llama-quantize` memory-maps its input
and works one tensor at a time.

| Size | Label | RAM | Disk |
|---|---|---|---|
| 2vcpu | `blacksmith-2vcpu-ubuntu-2404-arm` | 6 GB | 75 GB |
| 4vcpu | `blacksmith-4vcpu-ubuntu-2404-arm` | 12 GB | 75 GB |
| 8vcpu | `blacksmith-8vcpu-ubuntu-2404-arm` | 24 GB | 160 GB |
| 16vcpu | `blacksmith-16vcpu-ubuntu-2404-arm` | 48 GB | 750 GB |
| 32vcpu | `blacksmith-32vcpu-ubuntu-2404-arm` | 96 GB | 1.5 TB |

The safetensors pipeline uses at least `4vcpu`, and GGUF repos usually fit on `2vcpu`. With the default quant
list and `target_minutes: 15`, auto picks:
- Qwen3-0.6B: `4vcpu` (~4 min)
- granite-4.2-3b: `8vcpu` (~11 min)
- Qwen3-8B: `32vcpu` (~11 min; `16vcpu` would take ~16 min, so `target_minutes: 20` keeps it there)
- granite-4.2-30b: `32vcpu` (~39 min)

F32 is the biggest output and raises the peak disk need the most. With a BF16 source over about 12 B parameters,
the F32 file is split after quantizing, so it briefly needs twice its own size.

## Setup

1. Install the [Blacksmith](https://blacksmith.sh) GitHub App on the org and give it access to this repo.
2. Add a repository secret `HF_TOKEN` with **write** access. It is also used to download gated models, so
   accept their licenses with the same HF account first.
3. Optional: add a repository variable `HF_NAMESPACE` to upload to an org instead of the token owner.

## Running locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install 'huggingface_hub[hf_xet]' -r /path/to/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
cd scripts
python plan.py --model Qwen/Qwen3-0.6B --quants Q8_0,Q4_K_M --namespace me
python pipeline_safetensors.py --model Qwen/Qwen3-0.6B --out-repo me/Qwen3-0.6B-BE-GGUF \
  --quants BF16,Q8_0,Q4_K_M --llama-cpp /path/to/llama.cpp --bin /path/to/llama.cpp/build/bin \
  --work /tmp/work --no-upload
```

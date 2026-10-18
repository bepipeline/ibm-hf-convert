# ibm-hf-convert

GitHub Actions pipeline that turns a Hugging Face model into **big-endian** GGUF files (for s390x / IBM Z)
and uploads them to `<namespace>/<model>-BE-GGUF` on Hugging Face. It runs on Blacksmith ARM runners.

## Usage

Go to **Actions → Convert to big-endian GGUF → Run workflow**, then enter a model tag:

| Input | Default | Notes |
|---|---|---|
| `model` | — | `Qwen/Qwen3-8B` (safetensors) or `unsloth/Qwen3-8B-GGUF` (GGUF). Add `:Q4_K_M` to a GGUF repo to convert only that quant. |
| `quants` | `F32,F16,BF16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,IQ4_NL,IQ4_XS,MXFP4_MOE` | Safetensors models only. |
| `hf_namespace` | `HF_NAMESPACE` variable, else the token owner | User or org to upload to. |
| `runner` | `auto` | Sets the runner size. `auto` picks one as described in [Runner sizing](#runner-sizing). |
| `target_minutes` | `15` | Used by `auto`: the time budget for quantizing. Raise it to use smaller, cheaper runners. |
| `private` | `false` | Applies only when the output repo is created. |
| `llama_cpp_ref` | `latest` | `latest` or a release tag like `b11718` uses the prebuilt ARM tools from that release. A branch or commit is built from source, which takes about 2.5 min longer. |

### What happens

1. **Plan:** a job on GitHub's free `ubuntu-slim` runner reads the repo's file list from the HF API.
   - If the repo has `*.safetensors`, it runs the safetensors pipeline. If it has only `*.gguf`, it runs the GGUF pipeline.
   - It checks the requested quant types. It drops `MXFP4_MOE` for dense models, where the output would be a copy of `Q8_0`.
   - It estimates disk need and picks the smallest runner with enough disk.
2. **Safetensors pipeline:**
   1. Download the model.
   2. If any source tensor is not BF16, make F32/F16 directly from the safetensors.
   3. Make a little-endian BF16 intermediate, then delete the safetensors.
   4. Run `llama-quantize` once per remaining type. If every source tensor is BF16, this includes F32 and F16:
      the BF16 intermediate holds the source weights exactly, and `llama-quantize` is much faster than another
      `convert_hf_to_gguf.py` pass, which is mostly single-threaded and takes about 39 s per billion parameters.
      Outputs over 48 GB are split with `llama-gguf-split`.
   5. Byte-swap each output to big-endian, upload it and delete it locally. This runs on a background thread
      while the next output is converted or quantized. A new output starts only when the disk has room for it.
   6. Byte-swap and upload BF16 last, because the quant steps need it in little-endian.
3. **GGUF pipeline:** handles every `*.gguf` in the repo, including mmproj files and split shards, one at a time:
   1. Download the file.
   2. Byte-swap it to big-endian.
   3. Upload it as `<Model>-BE-<QUANT>.gguf` and delete it locally. The next file downloads while this one uploads.

   Files with tensor types the endian script can't handle (such as IQ2/IQ3 quants) are skipped with a warning.

Byte-swapping uses llama.cpp's `gguf-py/gguf/scripts/gguf_convert_endian.py`. Every output is checked to be
big-endian before upload. Files are named `<Model>-BE-<QUANT>.gguf` (for example `Bonsai-8B-BE-Q1_0.gguf`).
Split outputs go under `<QUANT>/`.

At the end of every run, the repo's `README.md` is regenerated in one commit:
- **Front matter:** the original card's front matter, with `base_model` pointing at the source and these
  tags added: `gguf`, `mainframe`, `s390x`, `z15`, `z16`, `z17`, `big-endian`.
- **Header:** creator, original model, a description and a **Provided Files** table listing every GGUF in the
  repo, including files from earlier runs.
- **Original card:** the source model's card under `# Original model card: …`.

The original's LICENSE/NOTICE files, images (plus any file the card links to by relative path) and
`.eval_results/` are copied too, so card images still render.
Any manual edits to the README are overwritten on the next run.

A final **Save model card** job on `ubuntu-slim` also commits the same card to this repo as
`models/<Model>/README.md`, for example `models/Bonsai-8B/README.md`. The copied images and license files go next to it, so
the GitHub copy renders the same way. It runs even if some quants failed, since the card
still lists everything that was uploaded. It needs push access to the branch the workflow ran from, so if
that branch is protected, allow `github-actions[bot]` to push to it.

### Runner sizing

`auto` picks the smallest runner that meets two conditions:

1. **Disk:** it has room for everything that exists at the same time: the safetensors, the BF16 intermediate and
   the current outputs. Not enough disk is the one thing that can make a run fail.
2. **CPU time:** its estimated quantize time fits within `target_minutes`. Quantizing is CPU-bound: a
   granite-4.2-3b run on `4vcpu` sat at 99% CPU and about 15% memory. Runners are billed per vCPU-minute, so
   for that part more vCPUs cost about the same and finish sooner. The estimate uses each quant type's cost
   measured on that run (`QUANT_CPU_COST` in `scripts/common.py`). If no runner meets the target, the largest
   runner with enough disk is used.

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
- Qwen3-0.6B: `4vcpu` (~3 min)
- granite-4.2-3b: `8vcpu` (~8 min)
- Qwen3-8B: `16vcpu` (~9 min)
- 30B+ models: `32vcpu`

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
python plan.py --model Qwen/Qwen3-0.6B --quants Q8_0,Q4_K --namespace me
python pipeline_safetensors.py --model Qwen/Qwen3-0.6B --out-repo me/Qwen3-0.6B-BE-GGUF \
  --quants BF16,Q8_0,Q4_K --llama-cpp /path/to/llama.cpp --bin /path/to/llama.cpp/build/bin \
  --work /tmp/work --no-upload
```

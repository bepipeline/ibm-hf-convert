#!/usr/bin/env python3
"""Inspect a Hugging Face model and decide pipeline, runner size and output repo.

Inputs come from environment variables (set by the workflow) or CLI flags.
Outputs are written to $GITHUB_OUTPUT, or printed when run locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

from common import (
    CONVERTER_TYPES, OS_RESERVE, QUANT_SIZE_RATIO, RUNNERS, SPLIT_THRESHOLD,
    annotate, fmt_bytes, gh_output, gh_summary, is_moe, model_name, output_repo,
    parse_model_tag, parse_quants, runner_label, select_gguf_files, select_safetensors_files,
)
from card import base_name

SAFETENSORS_MIN_RUNNER = "4vcpu"  # 12 GB RAM floor for the converter and large embedding tensors
GGUF_MIN_RUNNER = "2vcpu"


def safetensors_disk_need(st_bytes: int, bf16_bytes: int, quants: list[str]) -> int:
    # Phase A: safetensors on disk + the biggest file made straight from them (F32/F16 or the BF16 intermediate).
    phase_a = st_bytes + (2 * bf16_bytes if "F32" in quants else bf16_bytes)
    # Phase B: BF16 intermediate + largest quant (+ a second copy while llama-gguf-split runs).
    quantized = [q for q in quants if q not in CONVERTER_TYPES and q != "BF16"]
    phase_b = 0
    if quantized:
        out = max(QUANT_SIZE_RATIO[q] for q in quantized) * bf16_bytes
        phase_b = bf16_bytes + (2 * out if out > SPLIT_THRESHOLD else out)
    return int(max(phase_a, phase_b) * 1.1) + OS_RESERVE


def pick_runner(need: int, minimum: str, override: str) -> str:
    sizes = [r[0] for r in RUNNERS]
    if override and override != "auto":
        if override not in sizes:
            raise SystemExit(f"Unknown runner size {override!r}; choose one of {sizes}")
        disk = RUNNERS[sizes.index(override)][3]
        if disk < need:
            annotate("warning", f"Runner {override} has {disk // 10**9} GB disk but ~{fmt_bytes(need)} is estimated; the job may run out of space")
        return override
    for size, _, _, disk in RUNNERS[sizes.index(minimum):]:
        if disk >= need:
            return size
    raise SystemExit(
        f"Estimated disk need {fmt_bytes(need)} exceeds the largest Blacksmith ARM runner "
        f"({RUNNERS[-1][3] // 10**9} GB). Drop F32 from the quant list or convert fewer files."
    )


def resolve_namespace(api: HfApi, explicit: str) -> str:
    if explicit:
        return explicit
    if os.environ.get("HF_NAMESPACE"):
        return os.environ["HF_NAMESPACE"]
    try:
        return api.whoami()["name"]
    except Exception as e:
        raise SystemExit(f"Could not determine HF namespace (set hf_namespace or a valid HF_TOKEN): {e}") from None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("INPUT_MODEL", ""))
    p.add_argument("--quants", default=os.environ.get("INPUT_QUANTS", ""))
    p.add_argument("--namespace", default=os.environ.get("INPUT_HF_NAMESPACE", ""))
    p.add_argument("--runner", default=os.environ.get("INPUT_RUNNER", "auto"))
    args = p.parse_args()

    api = HfApi()
    repo_id, quant_tag = parse_model_tag(args.model)
    info = api.model_info(repo_id, files_metadata=True)
    siblings = info.siblings or []
    has_st = any(s.rfilename.endswith(".safetensors") for s in siblings)
    has_gguf = any(s.rfilename.endswith(".gguf") for s in siblings)

    namespace = resolve_namespace(api, args.namespace.strip())
    out_repo = output_repo(namespace, repo_id)
    rows = [("Source", f"[{repo_id}](https://huggingface.co/{repo_id})"), ("Output repo", f"[{out_repo}](https://huggingface.co/{out_repo})")]

    if has_st and not quant_tag:
        pipeline = "safetensors"
        files = select_safetensors_files(siblings)
        st_bytes = sum(size for name, size in files if name.endswith(".safetensors"))
        params = info.safetensors.total if info.safetensors else 0
        bf16_bytes = 2 * params if params else st_bytes
        quants = parse_quants(args.quants)
        if "MXFP4_MOE" in quants:
            config = json.load(open(hf_hub_download(repo_id, "config.json")))
            if not is_moe(config):
                quants.remove("MXFP4_MOE")
                annotate("notice", "Dropping MXFP4_MOE: dense model (it would only duplicate Q8_0)")
        if not quants:
            raise SystemExit("Nothing left to produce")
        need = safetensors_disk_need(st_bytes, bf16_bytes, quants)
        size = pick_runner(need, SAFETENSORS_MIN_RUNNER, args.runner)
        rows += [
            ("Pipeline", "safetensors → GGUF → quantize → big-endian"),
            ("Safetensors", fmt_bytes(st_bytes)),
            ("Parameters", f"{params / 1e9:.2f} B" if params else "unknown (estimated from file size)"),
            ("Quants", ", ".join(quants)),
        ]
    elif has_gguf:
        pipeline = "gguf"
        files = select_gguf_files(siblings, quant_tag)
        if not files:
            raise SystemExit(f"No GGUF files in {repo_id} match tag {quant_tag!r}")
        quants = []
        largest = max(size for _, size in files)
        need = int(largest * 1.05) + OS_RESERVE  # one file at a time, byte-swapped in place
        size = pick_runner(need, GGUF_MIN_RUNNER, args.runner)
        rows += [
            ("Pipeline", "GGUF → big-endian"),
            ("Files", f"{len(files)} ({fmt_bytes(sum(s for _, s in files))} total, largest {fmt_bytes(largest)})"),
        ]
    else:
        raise SystemExit(f"{repo_id} has no *.safetensors or *.gguf files")

    label = runner_label(size)
    rows += [("Estimated disk need", fmt_bytes(need)), ("Runner", f"`{label}`")]
    gh_summary("### Plan\n\n| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows))
    gh_output(
        pipeline=pipeline,
        repo_id=repo_id,
        quant_tag=quant_tag or "",
        model_name=model_name(repo_id),
        base_name=base_name(repo_id),
        out_repo=out_repo,
        quants=",".join(quants),
        runner=label,
        need_bytes=need,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, SystemExit) as e:
        msg = e.code if isinstance(e, SystemExit) else str(e)
        if not isinstance(msg, str):
            raise
        annotate("error", msg)
        sys.exit(1)

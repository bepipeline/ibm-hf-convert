#!/usr/bin/env python3
"""safetensors -> GGUF (F32/F16/BF16 + llama-quantize types) -> big-endian -> Hugging Face.

Disk-frugal ordering:
  1. download safetensors
  2. F32/F16 straight from safetensors; each is byte-swapped, uploaded and deleted
  3. little-endian BF16 intermediate; safetensors deleted
  4. each quant from the BF16 intermediate; split if > 48 GB, byte-swapped, uploaded, deleted
  5. BF16 intermediate byte-swapped and uploaded last (it must stay little-endian for quantizing)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from common import (
    CONVERTER_TYPES, SPLIT_MAX_SIZE, SPLIT_THRESHOLD, Endian, Results, Uploader,
    check_free_disk, fmt_bytes, group, parse_quants, run, select_safetensors_files,
)
from card import base_name


def shards(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.gguf"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="source repo id, e.g. Qwen/Qwen3-8B")
    p.add_argument("--out-repo", required=True)
    p.add_argument("--quants", required=True, help="comma-separated, e.g. BF16,Q8_0,Q4_K")
    p.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout")
    p.add_argument("--bin", type=Path, required=True, help="dir with llama-quantize and llama-gguf-split")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--need-bytes", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--private", action="store_true", help="create the output repo as private")
    p.add_argument("--no-upload", action="store_true", help="keep files locally, don't touch HF")
    args = p.parse_args()

    name = base_name(args.model)
    quants = parse_quants(args.quants)
    direct = [q for q in quants if q in CONVERTER_TYPES]
    quantized = [q for q in quants if q not in CONVERTER_TYPES and q != "BF16"]
    need_bf16 = bool(quantized) or "BF16" in quants

    work: Path = args.work.resolve()
    src, out_root, bf16_dir = work / "src", work / "out", work / "bf16"
    convert = args.llama_cpp / "convert_hf_to_gguf.py"
    quantize = args.bin / "llama-quantize"
    gguf_split = args.bin / "llama-gguf-split"
    endian = Endian(args.llama_cpp)
    uploader = Uploader(args.out_repo, args.model, enabled=not args.no_upload, private=args.private)
    results = Results()

    if args.need_bytes:
        check_free_disk(work, args.need_bytes)

    def publish(files: list[Path], qtype: str) -> None:
        """Byte-swap, upload and delete one output (single file or split shards)."""
        single = len(files) == 1
        total = sum(f.stat().st_size for f in files)
        for f in files:
            endian.to_big(f)
            uploader.upload(f, f.name if single else f"{qtype}/{f.name}")
            if not args.no_upload:
                f.unlink()
        results.ok(qtype, f"{len(files)} file(s), {fmt_bytes(total)}")

    with group(f"Download {args.model}"):
        files = [f for f, _ in select_safetensors_files(HfApi().model_info(args.model, files_metadata=True).siblings)]
        snapshot_download(args.model, local_dir=src, allow_patterns=files)

    uploader.ensure_repo()

    # 2. F32 / F16 straight from safetensors
    for qtype in direct:
        out_dir = out_root / qtype
        try:
            with group(f"{qtype}: convert"):
                out_dir.mkdir(parents=True, exist_ok=True)
                run([sys.executable, convert, src, "--outtype", qtype.lower(),
                     "--outfile", out_dir / f"{name}-BE.{qtype}.gguf", "--split-max-size", SPLIT_MAX_SIZE])
            with group(f"{qtype}: big-endian + upload"):
                publish(shards(out_dir), qtype)
        except Exception as e:
            results.fail(qtype, e)
        finally:
            if not args.no_upload:
                shutil.rmtree(out_dir, ignore_errors=True)

    # 3. BF16 intermediate (little-endian, input for llama-quantize)
    intermediate: Path | None = None
    if need_bf16:
        try:
            with group("BF16 intermediate: convert"):
                bf16_dir.mkdir(parents=True, exist_ok=True)
                run([sys.executable, convert, src, "--outtype", "bf16",
                     "--outfile", bf16_dir / f"{name}-BE.BF16.gguf", "--split-max-size", SPLIT_MAX_SIZE])
                intermediate = shards(bf16_dir)[0]  # llama-quantize finds the other shards itself
        except Exception as e:
            for qtype in quantized + (["BF16"] if "BF16" in quants else []):
                results.fail(qtype, RuntimeError(f"BF16 intermediate failed: {e}"))
    shutil.rmtree(src, ignore_errors=True)

    # 4. llama-quantize types
    if intermediate:
        for qtype in quantized:
            out_dir = out_root / qtype
            try:
                with group(f"{qtype}: quantize"):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out = out_dir / f"{name}-BE.{qtype}.gguf"
                    run([quantize, intermediate, out, qtype, args.threads])
                    if out.stat().st_size > SPLIT_THRESHOLD:
                        run([gguf_split, "--split", "--split-max-size", SPLIT_MAX_SIZE, out, out_dir / f"{name}-BE.{qtype}"])
                        out.unlink()
                with group(f"{qtype}: big-endian + upload"):
                    publish(shards(out_dir), qtype)
            except Exception as e:
                results.fail(qtype, e)
            finally:
                if not args.no_upload:
                    shutil.rmtree(out_dir, ignore_errors=True)

        # 5. BF16 last: byte-swapping destroys it as a quantize input
        if "BF16" in quants:
            try:
                with group("BF16: big-endian + upload"):
                    publish(shards(bf16_dir), "BF16")
            except Exception as e:
                results.fail("BF16", e)
        if not args.no_upload:
            shutil.rmtree(bf16_dir, ignore_errors=True)

    try:
        with group("Model card"):
            uploader.publish_card(work / "card")
    except Exception as e:
        results.fail("README.md", e)

    results.write_summary(f"{args.model} → {args.out_repo}")
    if results.failed:
        sys.exit(f"{results.failed} output(s) failed")


if __name__ == "__main__":
    main()

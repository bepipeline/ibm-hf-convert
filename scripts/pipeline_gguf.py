#!/usr/bin/env python3
"""Every GGUF in a Hugging Face repo -> big-endian -> Hugging Face, one file at a time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from common import (
    Endian, Results, UnsupportedTensorType, Uploader, check_free_disk, fmt_bytes, group, select_gguf_files,
)
from card import base_name, be_filename


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="source repo id, e.g. unsloth/Qwen3-8B-GGUF")
    p.add_argument("--tag", default="", help="only files matching this quant tag, e.g. Q4_K_M")
    p.add_argument("--out-repo", required=True)
    p.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout (gguf-py is enough)")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--need-bytes", type=int, default=0)
    p.add_argument("--private", action="store_true", help="create the output repo as private")
    p.add_argument("--no-upload", action="store_true", help="keep files locally, don't touch HF")
    args = p.parse_args()

    work: Path = args.work.resolve()
    base = base_name(args.model)
    endian = Endian(args.llama_cpp)
    uploader = Uploader(args.out_repo, args.model, enabled=not args.no_upload, private=args.private)
    results = Results()

    if args.need_bytes:
        check_free_disk(work, args.need_bytes)

    files = select_gguf_files(HfApi().model_info(args.model, files_metadata=True).siblings, args.tag or None)
    if not files:
        sys.exit(f"No GGUF files in {args.model}" + (f" match {args.tag!r}" if args.tag else ""))
    print(f"{len(files)} GGUF file(s) to convert")

    uploader.ensure_repo()

    for rel, size in files:
        path = work / rel
        try:
            with group(f"{rel} ({fmt_bytes(size)})"):
                hf_hub_download(args.model, rel, local_dir=work)
                try:
                    endian.check(path)
                except UnsupportedTensorType as e:
                    results.skip(rel, str(e))
                    continue
                endian.to_big(path)
                target = be_filename(base, rel)
                uploader.upload(path, target)
                results.ok(rel, f"→ {target}, {fmt_bytes(size)}")
        except Exception as e:
            results.fail(rel, e)
        finally:
            if not args.no_upload:
                path.unlink(missing_ok=True)

    try:
        with group("Model card"):
            uploader.publish_card(work / "card")
    except Exception as e:
        results.fail("README.md", e)

    results.write_summary(f"{args.model} → {args.out_repo}")
    if results.failed:
        sys.exit(f"{results.failed} file(s) failed")


if __name__ == "__main__":
    main()

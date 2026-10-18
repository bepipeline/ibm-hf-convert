#!/usr/bin/env python3
"""safetensors -> GGUF (F32/F16/BF16 + llama-quantize types) -> big-endian -> Hugging Face.

Disk-frugal ordering:
  1. download safetensors
  2. F32/F16 straight from safetensors, unless every source tensor is BF16
  3. little-endian BF16 intermediate; safetensors deleted
  4. each quant from the BF16 intermediate (F32/F16 too for a BF16 source); split if > 48 GB
  5. BF16 intermediate published last (it must stay little-endian for quantizing)

Every output is byte-swapped, uploaded and deleted on a background thread while the next one is
produced, as long as the disk has room for both.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader
from huggingface_hub import HfApi, snapshot_download

from common import (
    CONVERTER_TYPES, GIB, QUANT_SIZE_RATIO, SPLIT_MAX_SIZE, SPLIT_THRESHOLD, Endian, Publisher, Results, Uploader,
    adapt_tokenizer_config, annotate, check_free_disk, converter_tags_override, has_projector, fmt_bytes, group, parse_quants, resolve_native,
    is_mistral_format, run, select_safetensors_files,
)
from card import base_name


def shards(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.gguf"))


def converter_type_mismatches(intermediate: list[Path], out_files: list[Path], qtype: str) -> list[str]:
    """Tensors in an F32/F16 made by llama-quantize whose type differs from convert_hf_to_gguf.py --outtype <qtype>.

    The converter writes a tensor as BF16 in the intermediate exactly when it would write it as F32/F16 there, but
    llama-quantize skips some architecture-specific tensors and converts some the converter keeps in F32.
    """
    target = GGMLQuantizationType[qtype]
    want = {t.name: target if t.tensor_type == GGMLQuantizationType.BF16 else t.tensor_type
            for f in intermediate for t in GGUFReader(f).tensors}
    got = {t.name: t.tensor_type.name for f in out_files for t in GGUFReader(f).tensors}
    return [f"{n} is {got.get(n, 'missing')}, converter writes {w.name}" for n, w in want.items() if got.get(n) != w.name]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="source repo id, e.g. Qwen/Qwen3-8B")
    p.add_argument("--out-repo", required=True)
    p.add_argument("--quants", required=True, help="comma-separated, e.g. BF16,Q8_0,Q4_K_M")
    p.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout")
    p.add_argument("--bin", type=Path, required=True, help="dir with llama-quantize and llama-gguf-split")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--need-bytes", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--private", action="store_true", help="create the output repo as private")
    p.add_argument("--no-upload", action="store_true", help="keep files locally, don't touch HF")
    args = p.parse_args()

    name = base_name(args.model)

    work: Path = args.work.resolve()
    src, out_root, bf16_dir = work / "src", work / "out", work / "bf16"
    convert = args.llama_cpp / "convert_hf_to_gguf.py"
    quantize = args.bin / "llama-quantize"
    gguf_split = args.bin / "llama-gguf-split"
    endian = Endian(args.llama_cpp)
    uploader = Uploader(args.out_repo, args.model, enabled=not args.no_upload, private=args.private)
    publisher = Publisher(work)
    results = Results()

    if args.need_bytes:
        check_free_disk(work, args.need_bytes)

    info = HfApi().model_info(args.model, files_metadata=True)
    quants = resolve_native(parse_quants(args.quants), info.safetensors)
    files = select_safetensors_files(info.siblings)
    # Converter flags shared by every convert_hf_to_gguf.py call
    src_format = ["--mistral-format"] if is_mistral_format(info.siblings) else []
    bf16_bytes = 2 * info.safetensors.total if info.safetensors else sum(s for f, s in files if f.endswith(".safetensors"))
    # A BF16-only source survives the BF16 intermediate losslessly, so llama-quantize can make F32/F16 from it,
    # which is far faster than a mostly single-threaded converter pass each.
    bf16_source = bool(info.safetensors) and set(info.safetensors.parameters) == {"BF16"}
    direct = [] if bf16_source else [q for q in quants if q in CONVERTER_TYPES]
    quantized = [q for q in quants if q not in direct and q != "BF16"]
    need_bf16 = bool(quantized) or "BF16" in quants

    def publish(out_files: list[Path], qtype: str, out_dir: Path) -> None:
        """Byte-swap, upload and delete one output (single file or split shards). Runs on the publisher thread."""
        try:
            single = len(out_files) == 1
            total = sum(f.stat().st_size for f in out_files)
            for f in out_files:
                endian.to_big(f)
                uploader.upload(f, f.name if single else f"{qtype}/{f.name}")
                if not args.no_upload:
                    f.unlink()
            results.ok(qtype, f"{len(out_files)} file(s), {fmt_bytes(total)}")
        except Exception as e:
            results.fail(qtype, e)
        finally:
            if not args.no_upload:
                shutil.rmtree(out_dir, ignore_errors=True)

    def publish_copy(files: list[Path], qtype: str, out_dir: Path, copied: threading.Event) -> None:
        """Copy little-endian files that are still being read (the BF16 intermediate), then publish the copies."""
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            copies = []
            for f in files:
                shutil.copyfile(f, out_dir / f.name)
                copies.append(out_dir / f.name)
        except Exception as e:
            results.fail(qtype, e)
            shutil.rmtree(out_dir, ignore_errors=True)
            return
        finally:
            copied.set()
        publish(copies, qtype, out_dir)

    with group(f"Download {args.model}"):
        snapshot_download(args.model, local_dir=src, allow_patterns=[f for f, _ in files])
        for change in adapt_tokenizer_config(src):
            annotate("notice", f"tokenizer_config.json written by transformers 5, adapted for the converter: {change}")

    uploader.ensure_repo()
    # general.tags in every GGUF: the model's own tags plus the uploading account (llama-quantize copies it on)
    meta = converter_tags_override(src, [args.out_repo.split("/", 1)[0]], work / "metadata.json")

    # 2. F32 / F16 straight from safetensors (not BF16 sources)
    for qtype in direct:
        out_dir = out_root / qtype
        publisher.wait_for_room(QUANT_SIZE_RATIO[qtype] * bf16_bytes)
        try:
            with group(f"{qtype}: convert"):
                out_dir.mkdir(parents=True, exist_ok=True)
                run([sys.executable, convert, src, *src_format, "--outtype", qtype.lower(),
                     "--outfile", out_dir / f"{name}-BE-{qtype}.gguf", "--split-max-size", SPLIT_MAX_SIZE, "--metadata", meta])
        except Exception as e:
            results.fail(qtype, e)
            shutil.rmtree(out_dir, ignore_errors=True)
            continue
        publisher.submit(publish, shards(out_dir), qtype, out_dir)

    # 3. BF16 intermediate (little-endian, input for llama-quantize)
    intermediate: Path | None = None
    if need_bf16:
        publisher.wait_for_room(bf16_bytes)
        try:
            with group("BF16 intermediate: convert"):
                bf16_dir.mkdir(parents=True, exist_ok=True)
                run([sys.executable, convert, src, *src_format, "--outtype", "bf16",
                     "--outfile", bf16_dir / f"{name}-BE-BF16.gguf", "--split-max-size", SPLIT_MAX_SIZE, "--metadata", meta])
                intermediate = shards(bf16_dir)[0]  # llama-quantize finds the other shards itself
                bf16_bytes = sum(f.stat().st_size for f in shards(bf16_dir))
        except Exception as e:
            for qtype in quantized + (["BF16"] if "BF16" in quants else []):
                results.fail(qtype, RuntimeError(f"BF16 intermediate failed: {e}"))

    # Vision/audio projector (small): one F16 mmproj, named so HF's `-hf` manifest pairs it with the model
    configs = [json.loads((src / f).read_text()) for f in ("config.json", "params.json") if (src / f).exists()]
    if any(has_projector(c) for c in configs):
        out_dir = out_root / "mmproj"
        publisher.wait_for_room(2 * GIB)
        try:
            with group("mmproj-F16: convert"):
                out_dir.mkdir(parents=True, exist_ok=True)
                run([sys.executable, convert, src, *src_format, "--mmproj", "--outtype", "f16",
                     "--outfile", out_dir / f"mmproj-{name}-BE-F16.gguf", "--metadata", meta])
        except Exception as e:
            results.skip("mmproj-F16", f"convert_hf_to_gguf.py --mmproj failed, architecture may be unsupported ({e})")
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            publisher.submit(publish, shards(out_dir), "mmproj-F16", out_dir)
    shutil.rmtree(src, ignore_errors=True)

    # BF16 output early, from a copy, so it uploads while quantizing runs instead of after it (8 min of a
    # 30B run's tail). Only when the disk has room for the copy next to a full queue of quants; otherwise
    # the intermediate itself is published once quantizing is done.
    bf16_copied: threading.Event | None = None
    if intermediate and "BF16" in quants:
        largest = max((QUANT_SIZE_RATIO[q] for q in quantized), default=0) * bf16_bytes
        if shutil.disk_usage(work).free > bf16_bytes + publisher.max_pending * largest + 20 * GIB:
            print("[publish] room to publish BF16 early from a copy", flush=True)
            bf16_copied = threading.Event()
            publisher.submit(publish_copy, shards(bf16_dir), "BF16", out_root / "BF16", bf16_copied)

    # 4. llama-quantize types (including F32 / F16 for BF16 sources)
    if intermediate:
        for qtype in quantized:
            out_dir = out_root / qtype
            est = QUANT_SIZE_RATIO[qtype] * bf16_bytes
            publisher.wait_for_room(2 * est if est > SPLIT_THRESHOLD else est)
            try:
                with group(f"{qtype}: quantize"):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out = out_dir / f"{name}-BE-{qtype}.gguf"
                    run([quantize, intermediate, out, qtype, args.threads])
                    if out.stat().st_size > SPLIT_THRESHOLD:
                        run([gguf_split, "--split", "--split-max-size", SPLIT_MAX_SIZE, out, out_dir / f"{name}-BE-{qtype}"])
                        out.unlink()
                    if qtype in CONVERTER_TYPES and (diff := converter_type_mismatches(shards(bf16_dir), shards(out_dir), qtype)):
                        annotate("warning", f"{qtype}: {len(diff)} tensor type(s) differ from convert_hf_to_gguf.py: {'; '.join(diff[:5])}")
            except Exception as e:
                results.fail(qtype, e)
                shutil.rmtree(out_dir, ignore_errors=True)
                continue
            publisher.submit(publish, shards(out_dir), qtype, out_dir)

        # 5. BF16 last, unless already published from a copy: byte-swapping destroys it as a quantize input
        if "BF16" in quants and not bf16_copied:
            publisher.submit(publish, shards(bf16_dir), "BF16", bf16_dir)
        elif not args.no_upload:
            if bf16_copied:
                bf16_copied.wait()  # the copy may still be reading the intermediate
            shutil.rmtree(bf16_dir, ignore_errors=True)

    with group("Waiting for uploads"):
        publisher.drain()

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

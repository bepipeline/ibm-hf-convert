"""Shared helpers for planning, converting, byte-swapping and uploading GGUF files."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import card

GB = 1000**3
GIB = 1024**3

# HF rejects/discourages files above 50 GB; llama.cpp tools parse "G" as 10^9 bytes.
SPLIT_MAX_SIZE = "48G"
SPLIT_THRESHOLD = 48 * GB

# Disk the runner image, Python env and llama.cpp build already occupy.
OS_RESERVE = 20 * GIB

# Blacksmith ARM runners: (size, vCPU, RAM bytes, disk bytes)
RUNNERS = [
    ("2vcpu", 2, 6 * GB, 75 * GB),
    ("4vcpu", 4, 12 * GB, 75 * GB),
    ("8vcpu", 8, 24 * GB, 160 * GB),
    ("16vcpu", 16, 48 * GB, 750 * GB),
    ("32vcpu", 32, 96 * GB, 1500 * GB),
]


def runner_label(size: str) -> str:
    return f"blacksmith-{size}-ubuntu-2404-arm"


# Output types that llama-quantize can make without an imatrix *and* that
# gguf_convert_endian.py can byte-swap (including every fallback type they may emit).
# Value: estimated output size relative to the BF16 GGUF, used for disk planning.
QUANT_SIZE_RATIO = {
    "F32": 2.0,
    "F16": 1.0,
    "BF16": 1.0,
    "Q8_0": 0.54,
    "MXFP4_MOE": 0.54,
    "Q6_K": 0.42,
    "Q5_1": 0.39,
    "Q5_K": 0.37,
    "Q5_K_M": 0.37,
    "Q5_K_S": 0.36,
    "Q5_0": 0.36,
    "Q4_1": 0.33,
    "Q4_K": 0.32,
    "Q4_K_M": 0.32,
    "Q4_K_S": 0.30,
    "Q4_0": 0.30,
    "IQ4_NL": 0.30,
    "IQ4_XS": 0.29,
    "Q3_K": 0.27,
    "Q3_K_L": 0.28,
    "Q3_K_M": 0.27,
    "Q3_K_S": 0.24,
    "Q2_K": 0.22,
    "Q2_0": 0.16,
    "TQ2_0": 0.16,
    "TQ1_0": 0.14,
    "Q1_0": 0.10,
}

# llama-quantize CPU cost in vCPU-seconds per billion parameters on Blacksmith ARM, measured from
# ibm-granite/granite-4.2-3b on a 4-vCPU runner (wall seconds x 4 / 3.66 B). Unmeasured types are estimates.
QUANT_CPU_COST = {
    "Q8_0": 17, "Q4_0": 18, "Q4_1": 15, "Q5_0": 15, "Q5_1": 16, "MXFP4_MOE": 20,
    "Q1_0": 18, "Q2_0": 18, "TQ1_0": 20, "TQ2_0": 20,
    "Q2_K": 69, "Q3_K": 52, "Q3_K_S": 52, "Q3_K_M": 52, "Q3_K_L": 56,
    "Q4_K": 101, "Q4_K_S": 101, "Q4_K_M": 101, "Q5_K": 86, "Q5_K_S": 86, "Q5_K_M": 86, "Q6_K": 56,
    "IQ4_NL": 246, "IQ4_XS": 250,
}
# Uploads hash on the CPU while the next quant runs, which slows quantizing a little.
PUBLISH_OVERHEAD = 1.15
# llama-quantize doesn't scale linearly with threads. Measured: 8 vCPU ~1.0 (granite-4.2-3b, estimate matched);
# 32 vCPU 0.64 for K-quants and 0.46 overall (granite-4.2-30b, with F32/F16 uploads competing for disk).
# 16 vCPU is interpolated.
THREAD_EFFICIENCY = {2: 1.0, 4: 1.0, 8: 1.0, 16: 0.75, 32: 0.55}

# Produced straight from safetensors by convert_hf_to_gguf.py instead of llama-quantize.
CONVERTER_TYPES = ("F32", "F16")

MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts", "num_routed_experts")

# Weight formats we never need when converting from safetensors.
SAFETENSORS_EXCLUDE = ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.onnx", "*.onnx_data", "*.h5", "*.msgpack", "*.tflite", "*.ot", "*.mlmodel")


# ---------------------------------------------------------------------------
# Model tags and repo naming
# ---------------------------------------------------------------------------

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_model_tag(tag: str) -> tuple[str, str | None]:
    """'org/name[:QUANT]' -> ('org/name', 'QUANT' | None)."""
    tag = tag.strip()
    tag = re.sub(r"^https?://huggingface\.co/", "", tag).strip("/")
    repo_id, _, quant = tag.partition(":")
    if not _REPO_RE.match(repo_id):
        raise ValueError(f"Invalid model tag {tag!r}; expected <org>/<model>[:<quant>]")
    return repo_id, (quant or None)


def model_name(repo_id: str) -> str:
    return repo_id.split("/", 1)[1]


def output_repo(namespace: str, repo_id: str) -> str:
    return f"{namespace}/{card.base_name(repo_id)}-BE-GGUF"


# llama-quantize aliases (quantize.cpp maps them to the _M ftype). Outputs are named by the full type,
# which is also the only form HF's `-hf repo:TAG` lookup accepts.
QUANT_ALIASES = {"Q3_K": "Q3_K_M", "Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M"}


# Stands for the source model's own float type, e.g. BF16 for a BF16-trained model. Converting a BF16 model to
# F32 only doubles the size, and to F16 loses exponent range; on a z17, BF16 also ran faster than F16.
NATIVE = "NATIVE"
FLOAT_TYPES = ("F32", "F16", "BF16")


def native_float_type(safetensors) -> str:
    """The float type holding most of the source's parameters; BF16 when that isn't a GGUF float type (FP8, int)."""
    params = dict(safetensors.parameters) if safetensors else {}
    dominant = max(params, key=params.get) if params else "BF16"
    return dominant if dominant in FLOAT_TYPES else "BF16"


def resolve_native(quants: list[str], safetensors) -> list[str]:
    """Replace NATIVE with the source's float type, keeping its place in the list and dropping duplicates."""
    out: list[str] = []
    for q in quants:
        q = native_float_type(safetensors) if q == NATIVE else q
        if q not in out:
            out.append(q)
    return out


def parse_quants(raw: str) -> list[str]:
    quants: list[str] = []
    for q in re.split(r"[\s,]+", raw.strip().upper()):
        q = QUANT_ALIASES.get(q, q)
        if q and q not in quants:
            quants.append(q)
    bad = [q for q in quants if q not in QUANT_SIZE_RATIO and q != NATIVE]
    if bad:
        raise ValueError(
            f"Unsupported quant type(s): {', '.join(bad)}. "
            f"Supported (no imatrix needed, endian-convertible): {NATIVE}, {', '.join(QUANT_SIZE_RATIO)}"
        )
    if not quants:
        raise ValueError("No quant types given")
    return quants


# ---------------------------------------------------------------------------
# Repo file selection (shared by plan.py and the pipelines so both agree)
# ---------------------------------------------------------------------------

def _matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def is_mistral_format(siblings) -> bool:
    """Mistral's native layout: params.json + consolidated*.safetensors + tekken.json / tokenizer.model*.

    llama.cpp converts it with --mistral-format (via mistral-common). Newer Mistral repos need this: their
    HF-format tokenizer only loads in transformers 5, newer than the one llama.cpp's converter pins.
    """
    names = [Path(s.rfilename).name for s in siblings if not s.rfilename.startswith("original/")]
    return ("params.json" in names
            and any(n.startswith("consolidated") and n.endswith(".safetensors") for n in names)
            and any(n == "tekken.json" or n.startswith("tokenizer.model") for n in names))


def select_safetensors_files(siblings) -> list[tuple[str, int]]:
    """Files needed by convert_hf_to_gguf.py: weights + config/tokenizer, minus other formats and duplicates.

    Mistral repos ship the weights twice (HF-format shards and consolidated*.safetensors); keep only the set
    matching the format we convert from.
    """
    mistral = is_mistral_format(siblings)
    selected = []
    for s in siblings:
        name, base = s.rfilename, Path(s.rfilename).name
        if name.startswith("original/") or _matches(base, SAFETENSORS_EXCLUDE):
            continue
        if base.endswith(".safetensors") and base.startswith("consolidated") != mistral:
            continue
        selected.append((name, s.size or 0))
    return selected


def select_gguf_files(siblings, quant_tag: str | None) -> list[tuple[str, int]]:
    # Byte-identical copies (e.g. 'Model.gguf' next to 'Model-Q1_0.gguf') are converted once,
    # keeping the longest name since it usually carries the quant type.
    by_hash: dict[str, tuple[str, int]] = {}
    for s in siblings:
        if not s.rfilename.endswith(".gguf"):
            continue
        key = s.lfs.sha256 if s.lfs else s.rfilename
        if key not in by_hash or len(s.rfilename) > len(by_hash[key][0]):
            by_hash[key] = (s.rfilename, s.size or 0)
    files = list(by_hash.values())
    if quant_tag:
        tag_re = re.compile(r"(?<![A-Za-z0-9])" + re.escape(quant_tag) + r"(?![A-Za-z0-9_])", re.IGNORECASE)
        files = [f for f in files if tag_re.search(f[0])]
    return sorted(files)


def has_projector(config: dict) -> bool:
    """Vision/audio models: convert_hf_to_gguf.py --mmproj exports their encoder + projector."""
    return any(config.get(k) for k in ("vision_config", "audio_config", "whisper_config", "vision_encoder"))


def is_later_shard(name: str) -> bool:
    """'x-00002-of-00003.gguf' -> True: shards after the first hold tensors only, no general.* metadata."""
    m = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name)
    return bool(m) and int(m.group(1)) > 1


def is_moe(config: dict) -> bool:
    for key, value in config.items():
        if key in MOE_CONFIG_KEYS and isinstance(value, int) and value > 1:
            return True
        if isinstance(value, dict) and is_moe(value):
            return True
    return False


# ---------------------------------------------------------------------------
# GitHub Actions helpers
# ---------------------------------------------------------------------------

def gh_output(**values) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in values.items()]
    if path:
        with open(path, "a") as fh:
            fh.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


def gh_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as fh:
            fh.write(markdown + "\n")
    else:
        print(markdown)


def annotate(level: str, message: str) -> None:
    """level: notice | warning | error"""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::{level}::{message}", flush=True)
    else:
        print(f"[{level}] {message}", flush=True)


@contextmanager
def group(title: str):
    print(f"::group::{title}" if os.environ.get("GITHUB_ACTIONS") else f"=== {title}", flush=True)
    try:
        yield
    finally:
        if os.environ.get("GITHUB_ACTIONS"):
            print("::endgroup::", flush=True)


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def check_free_disk(path: Path, need: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    print(f"Free disk at {path}: {fmt_bytes(free)} (planned need excl. OS reserve: {fmt_bytes(max(need - OS_RESERVE, 0))})")
    if free < need - OS_RESERVE:
        raise SystemExit(
            f"Not enough disk: {fmt_bytes(free)} free, need ~{fmt_bytes(need - OS_RESERVE)}. "
            "Pick a bigger runner or drop F32."
        )


# ---------------------------------------------------------------------------
# GGUF endianness
# ---------------------------------------------------------------------------

def gguf_endianness(path: Path) -> str:
    with open(path, "rb") as fh:
        magic = fh.read(4)
        (version,) = struct.unpack("<I", fh.read(4))
    if magic != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    # Versions are small numbers; read with the wrong byte order the low 16 bits are zero.
    return "BIG" if version & 0xFFFF == 0 else "LITTLE"


class UnsupportedTensorType(Exception):
    pass


@dataclass
class Endian:
    llama_cpp: Path

    @property
    def script(self) -> Path:
        return self.llama_cpp / "gguf-py" / "gguf" / "scripts" / "gguf_convert_endian.py"

    def _env(self) -> dict:
        return {**os.environ, "TQDM_DISABLE": "1"}

    def check(self, path: Path) -> None:
        """Raise UnsupportedTensorType if the endian script can't handle every tensor in the file."""
        proc = subprocess.run(
            [sys.executable, str(self.script), str(path), "big", "--dry-run"],
            capture_output=True, text=True, env=self._env(),
        )
        if proc.returncode != 0:
            out = proc.stdout + proc.stderr
            m = re.search(r"ValueError: (Cannot handle type .*)", out)
            if m:
                raise UnsupportedTensorType(m.group(1))
            print(out, file=sys.stderr)
            raise subprocess.CalledProcessError(proc.returncode, proc.args)

    def to_big(self, path: Path) -> None:
        if gguf_endianness(path) == "BIG":
            print(f"[publish] {path.name} is already big-endian", flush=True)
            return
        start = time.monotonic()
        # The script asks for confirmation on stdin and exits 0 even when it does nothing.
        # Output is captured because this runs on the publisher thread next to other logs.
        proc = subprocess.run(
            [sys.executable, str(self.script), str(path), "big"],
            input="YES\n", capture_output=True, text=True, env=self._env(),
        )
        if proc.returncode != 0:
            print(proc.stdout + proc.stderr, file=sys.stderr, flush=True)
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
        if gguf_endianness(path) != "BIG":
            raise RuntimeError(f"{path} is still little-endian after conversion")
        print(f"[publish] {path.name}: big-endian in {time.monotonic() - start:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Source fix-ups
# ---------------------------------------------------------------------------

def adapt_tokenizer_config(src: Path) -> list[str]:
    """Let llama.cpp's pinned transformers 4.x read a tokenizer_config.json written by transformers 5.

    v5 calls the tokenizer.json-backed tokenizer `TokenizersBackend` (v4: PreTrainedTokenizerFast) and
    stores extra special tokens as a list (v4: additional_special_tokens). Rewrites both in place and
    returns a description of each change.
    """
    path = src / "tokenizer_config.json"
    if not path.exists() or not (src / "tokenizer.json").exists():
        return []
    cfg = json.loads(path.read_text())
    changes = []
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        changes.append("tokenizer_class TokenizersBackend -> PreTrainedTokenizerFast")
    if isinstance(cfg.get("extra_special_tokens"), list):
        extra = cfg.pop("extra_special_tokens")
        known = list(cfg.get("additional_special_tokens") or [])
        cfg["additional_special_tokens"] = known + [t for t in extra if t not in known]
        changes.append(f"extra_special_tokens list ({len(extra)}) -> additional_special_tokens")
    if changes:
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return changes


# ---------------------------------------------------------------------------
# GGUF metadata tags
# ---------------------------------------------------------------------------

def converter_tags_override(src: Path, extra: list[str], out: Path) -> Path:
    """Metadata override for convert_hf_to_gguf.py --metadata: the general.tags the converter would derive
    on its own (model card tags + pipeline_tag, via the same gguf-py code) with `extra` appended."""
    import gguf

    try:
        tags = list(gguf.Metadata.load(None, src).tags or [])
    except Exception as e:
        annotate("warning", f"Could not read model card tags ({e}); general.tags will only hold {extra}")
        tags = []
    out.write_text(json.dumps({gguf.Keys.General.TAGS: tags + [t for t in extra if t not in tags]}))
    return out


def add_gguf_tags(path: Path, extra: list[str]) -> bool:
    """Append `extra` to general.tags of a little-endian GGUF by rewriting it (one full copy, so 2x its size
    on disk for a moment). Split shards after the first carry no general.* metadata and are left alone."""
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    reader = gguf.GGUFReader(path, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch is None:
        return False
    field = reader.get_field(gguf.Keys.General.TAGS)
    tags = list(field.contents()) if field else []
    missing = [t for t in extra if t not in tags]
    if not missing:
        return False
    tmp = path.with_name(path.name + ".tagged")
    writer = gguf.GGUFWriter(tmp, arch=arch.contents(), endianess=reader.endianess)
    if (alignment := reader.get_field(gguf.Keys.General.ALIGNMENT)) is not None:
        writer.data_alignment = alignment.contents()
    copy_with_new_metadata(reader, writer, {
        gguf.Keys.General.TAGS: MetadataDetails(gguf.GGUFValueType.ARRAY, tags + missing, sub_type=gguf.GGUFValueType.STRING),
    }, [])
    del reader
    tmp.replace(path)
    return True


# ---------------------------------------------------------------------------
# Hugging Face upload
# ---------------------------------------------------------------------------

@dataclass
class Uploader:
    out_repo: str
    source_repo: str
    enabled: bool = True
    private: bool = False
    api: object = field(init=False, default=None)
    uploaded: list[tuple[str, int]] = field(init=False, default_factory=list)

    def __post_init__(self):
        from huggingface_hub import HfApi

        self.api = HfApi()

    def ensure_repo(self) -> None:
        if not self.enabled:
            print(f"[no-upload] would create/use {self.out_repo}")
            return
        url = self.api.create_repo(self.out_repo, repo_type="model", private=self.private, exist_ok=True)
        print(f"Uploading to {url}")

    def upload(self, path: Path, path_in_repo: str) -> None:
        size = path.stat().st_size
        self.uploaded.append((path_in_repo, size))
        if not self.enabled:
            print(f"[no-upload] would upload {path.name} -> {self.out_repo}/{path_in_repo}")
            return
        for attempt in range(1, 4):
            try:
                start = time.monotonic()
                print(f"[publish] uploading {path_in_repo} ({fmt_bytes(size)})", flush=True)
                self.api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path_in_repo,
                    repo_id=self.out_repo,
                    commit_message=f"Add {path_in_repo} (big-endian)",
                )
                print(f"[publish] uploaded {path_in_repo} in {time.monotonic() - start:.0f}s", flush=True)
                return
            except Exception as e:  # network hiccups on multi-GB uploads
                if attempt == 3:
                    raise
                annotate("warning", f"Upload of {path_in_repo} failed (attempt {attempt}): {e}; retrying")
                time.sleep(30 * attempt)

    def publish_card(self, work: Path) -> None:
        """Regenerate README.md from every GGUF now in the repo (including earlier runs)."""
        from huggingface_hub.hf_api import RepoFile

        if self.enabled:
            files = [
                (f.path, f.size) for f in self.api.list_repo_tree(self.out_repo, recursive=True)
                if isinstance(f, RepoFile) and f.path.endswith(".gguf")
            ]
        else:
            files = self.uploaded
        card.publish_card(self.api, self.source_repo, self.out_repo, files, work, self.enabled)


# ---------------------------------------------------------------------------
# Background publishing
# ---------------------------------------------------------------------------

def default_publish_workers() -> int:
    """Parallel byte-swap + upload jobs. One kept up on 8 vCPU, but on a 32-vCPU 30B run quantizing waited
    ~23 min for a single worker (each swap runs at ~140 MB/s and each upload at ~250 MB/s)."""
    cpu = os.cpu_count() or 1
    return 1 if cpu < 16 else 2 if cpu < 32 else 3


class Publisher:
    """Runs byte-swap + upload jobs on background threads so the next output can be produced meanwhile.

    Before starting a new output, call wait_for_room(): it blocks until the queue is short and the disk
    has room for that output, so overlapping never needs more disk than the planner budgeted for.
    Jobs must handle their own errors (record them in Results).
    """

    def __init__(self, work: Path, workers: int | None = None, max_pending: int | None = None):
        self.work = work
        self.workers = workers or default_publish_workers()
        self.max_pending = max_pending or self.workers + 3
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="publish")
        self.pending: list[Future] = []
        print(f"[publish] {self.workers} worker(s), up to {self.max_pending} queued output(s)", flush=True)

    def submit(self, fn, *args) -> None:
        self.pending.append(self.pool.submit(fn, *args))

    def wait_for_room(self, next_bytes: float) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        while True:
            self.pending = [f for f in self.pending if not f.done()]
            full = len(self.pending) >= self.max_pending
            tight = shutil.disk_usage(self.work).free < next_bytes * 1.05 + 2 * GIB
            if not self.pending or not (full or tight):
                return
            print(f"[publish] waiting for {len(self.pending)} upload(s) to free {'queue' if full else 'disk'}", flush=True)
            wait(self.pending, return_when=FIRST_COMPLETED)

    def drain(self) -> None:
        wait(self.pending)
        self.pending = []
        self.pool.shutdown()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Results:
    """Per-output outcomes; written from the main thread and the publisher threads."""
    rows: list[tuple[str, str, str]] = field(default_factory=list)  # (item, status, detail)
    failed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def ok(self, item: str, detail: str = "") -> None:
        with self.lock:
            self.rows.append((item, "uploaded", detail))

    def skip(self, item: str, reason: str) -> None:
        annotate("warning", f"Skipped {item}: {reason}")
        with self.lock:
            self.rows.append((item, "skipped", reason))

    def fail(self, item: str, err: BaseException) -> None:
        annotate("error", f"{item} failed: {err}")
        with self.lock:
            self.failed += 1
            self.rows.append((item, "failed", str(err).splitlines()[0] if str(err) else type(err).__name__))

    def write_summary(self, title: str) -> None:
        lines = [f"### {title}", "", "| Item | Status | Detail |", "|---|---|---|"]
        lines += [f"| `{i}` | {s} | {d.replace('|', '/')} |" for i, s, d in self.rows]
        gh_summary("\n".join(lines))

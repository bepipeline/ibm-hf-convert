"""Model card (README.md) for the big-endian repo: our header + file table, then the original card."""

from __future__ import annotations

import fnmatch
import html
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

import yaml

S390X_GUIDE = "https://github.com/ggml-org/llama.cpp/blob/master/docs/build-s390x.md"

EXTRA_TAGS = ["gguf", "mainframe", "s390x", "z15", "z16", "z17", "big-endian"]

# Front-matter keys that describe the original repo, not this one.
DROP_KEYS = ("model-index", "widget", "inference")

# Small non-weight files copied from the source repo so licenses and card images keep working.
COPY_PATTERNS = (
    "LICENSE*", "LICENCE*", "NOTICE*", "USE_POLICY*", "COPYING*",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.webp", "*.pdf",
)
COPY_MAX_BYTES = 50 * 1000**2

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$")
MMPROJ_RE = re.compile(r"(?:^|[-_.])mmproj(?=$|[-_.])", re.IGNORECASE)
QUANT_TOKEN = re.compile(r"(?<![A-Z0-9])(I?Q\d+_[A-Z0-9_]*[A-Z0-9]|TQ\d_0|BF16|F16|F32|MXFP4(?:_MOE)?)(?![A-Z0-9])")

USE_CASES = [
    (r"^F32$", "full precision, largest"),
    (r"^(BF16|F16)$", "half precision, effectively lossless"),
    (r"^Q8_0$", "very large, extremely low quality loss"),
    (r"^Q6_K", "very large, extremely low quality loss"),
    (r"^Q5_K", "large, very low quality loss - recommended"),
    (r"^Q5_[01]$", "legacy format, prefer Q5_K_M"),
    (r"^Q4_K", "medium, balanced quality - recommended"),
    (r"^IQ4_", "small, good quality for its size"),
    (r"^Q4_[01]$", "legacy format, prefer Q4_K_M"),
    (r"^MXFP4", "MoE experts in MXFP4, other tensors Q8_0"),
    (r"^Q3_K", "very small, high quality loss"),
    (r"^Q2_K", "smallest K-quant, significant quality loss"),
    (r"^(Q1_0|Q2_0|TQ\d_0)$", "low-bit format for models trained for it"),
]


def base_name(repo_id: str) -> str:
    """Model name used for the repo, the files and the models/ folder, title-cased:
    'unsloth/Qwen3-8B-GGUF' -> 'Qwen3-8B', 'ibm-granite/granite-4.2-3b' -> 'Granite-4.2-3B'."""
    return title_case(re.sub(r"-GGUF$", "", repo_id.split("/", 1)[1], flags=re.IGNORECASE))


ACRONYMS = {"it": "IT", "vl": "VL", "moe": "MoE"}


def title_case(name: str) -> str:
    """'granite-4.2-3b' -> 'Granite-4.2-3B': capitalize each '-' part and upper-case size labels.

    Parts that already contain a capital ('DeepSeek', 'A3B') and version parts ('v0.3') are kept as they are.
    """
    def part(p: str) -> str:
        if any(c.isupper() for c in p) or re.fullmatch(r"v\d[\w.]*", p):
            return p
        if p in ACRONYMS:
            return ACRONYMS[p]
        if re.fullmatch(r"(\d+x)?\d+(\.\d+)?[kmbt]|[a-z]\d+(\.\d+)?[kmbt]", p):  # 3b, 0.6b, 8x7b, a3b
            return p.upper().replace("X", "x")
        return p[:1].upper() + p[1:]

    return "-".join(part(p) for p in name.split("-"))


def be_filename(base: str, rel: str, variant: str | None = None) -> str:
    """Big-endian file name in the '<base>-BE-<variant>.gguf' scheme, keeping folder and shard suffix.

    Without an explicit variant it is taken from the source name: 'Qwen3-8B-Q4_K_M.gguf' -> 'Q4_K_M'.
    Projectors keep an 'mmproj-' prefix, which is what HF's `-hf` manifest looks for:
    'mmproj-F16.gguf' -> 'mmproj-Qwen3-8B-BE-F16.gguf'.
    """
    p = PurePosixPath(rel)
    stem = p.name[: -len(".gguf")]
    shard = ""
    if m := SHARD_RE.search(stem):
        shard, stem = m.group(0), stem[: m.start()]
    mmproj = bool(MMPROJ_RE.search(stem))
    if mmproj:
        stem = MMPROJ_RE.sub("", stem, count=1).strip("-._") or "F16"
    if variant is None:
        if stem.lower() == base.lower():
            variant = ""
        elif stem.lower().startswith(base.lower()) and stem[len(base)] in "-._":
            variant = stem[len(base) + 1:]
        else:
            variant = stem
    name = f"{base}-BE-{variant}{shard}.gguf" if variant else f"{base}-BE{shard}.gguf"
    return str(p.with_name(("mmproj-" if mmproj else "") + name))


def quant_info(filename: str) -> tuple[str, str, str]:
    """(quant method, bits, use case) from a '<base>-BE-<variant>.gguf' name ('-BE.' from older runs too)."""
    stem = SHARD_RE.sub("", PurePosixPath(filename).name[: -len(".gguf")])
    m = re.search(r"-BE[-.](.+)$", stem)
    method = m.group(1) if m else ""
    m = QUANT_TOKEN.search(method.upper())  # 'UD-Q4_K_XL' -> 'Q4_K_XL', 'qat-Q4_0' -> 'Q4_0'
    token = m.group(1) if m else ""
    bits = re.search(r"\d+", token).group(0) if token else ""
    if MMPROJ_RE.search(stem):
        method = method if MMPROJ_RE.search(method) else f"mmproj-{method}"
        use = "multimodal projector"
    else:
        use = next((u for pat, u in USE_CASES if re.search(pat, token)), "") if token else ""
    return method, bits, use


def human_size(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit in ("B", "K") else f"{n:.1f}{unit}"
        n /= 1024
    return ""


def files_table(out_repo: str, files: list[tuple[str, int]]) -> str:
    """One row per GGUF; split shards collapse into one row linking to the first shard."""
    groups: dict[str, list[tuple[str, int]]] = {}
    for path, size in sorted(files):
        p = PurePosixPath(path)
        key = str(p.with_name(SHARD_RE.sub("", p.name[: -len(".gguf")]) + ".gguf"))
        groups.setdefault(key, []).append((path, size))

    rows = []
    for key, parts in groups.items():
        first = parts[0][0]
        method, bits, use = quant_info(key)
        label = PurePosixPath(key).name + (f" ({len(parts)} parts)" if len(parts) > 1 else "")
        link = f"https://huggingface.co/{out_repo}/blob/main/{first}"
        size = human_size(sum(s for _, s in parts))
        rows.append((int(bits) if bits else 999, label, f"| [{label}]({link}) | {method} | {bits} | {size} | {use} |"))
    rows.sort()
    header = "| Name | Quant Method | Bits | Size | Use Case |\n|---|---|---|---|---|"
    return "\n".join([header] + [r[2] for r in rows])


def creator_name(api, owner: str) -> str:
    for fetch in (api.get_organization_overview, api.get_user_overview):
        try:
            if fullname := getattr(fetch(owner), "fullname", None):
                return fullname
        except Exception:
            pass
    return owner


def build_card(api, source: str, out_repo: str, files: list[tuple[str, int]]) -> str:
    from huggingface_hub import ModelCard

    try:
        original = ModelCard.load(source)
        data, text = original.data.to_dict(), original.text
    except Exception:
        data, text = {}, ""

    owner = source.split("/", 1)[0]

    for key in list(data):
        if key in DROP_KEYS or key.startswith("extra_gated"):
            del data[key]
    if data.get("library_name") not in ("llama.cpp", "gguf"):
        data.pop("library_name", None)
    tags = list(data.get("tags") or [])
    data["tags"] = tags + [t for t in EXTRA_TAGS if t not in tags]
    data["base_model"] = [source]
    data["base_model_relation"] = "quantized"
    data["quantized_by"] = out_repo.split("/", 1)[0]

    front = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    tag = example_tag(files)
    return f"""---
{front}
---

# Read our How to [Run on IBM Z & LinuxONE Guide!]({S390X_GUIDE})

<div>
<p style="margin: 0 0 0px 0; margin-top: 0px;">
<em>These models are byte-swapped to run on IBM Z & LinuxONE systems.</em>
</p>

<ul style="margin: 0;">
<li>Model Creator: <a href="https://huggingface.co/{owner}">{html.escape(creator_name(api, owner))}</a></li>
<li>Original Model: <a href="https://huggingface.co/{source}">{source}</a></li>
</ul>
</div>

<br>

# Provided Files

Get started on your IBM Z & LinuxONE system

```sh
build/bin/llama-server -hf {out_repo}{f":{tag}" if tag else ""}
```

{files_table(out_repo, files)}

---

{text.strip()}
"""


def example_tag(files: list[tuple[str, int]]) -> str:
    """Quant to show in the llama-server example: Q4_K_M if present, else the closest to 4 bits."""
    methods = {quant_info(path)[0]: quant_info(path)[1] for path, _ in files}
    methods = {m: int(b) for m, b in methods.items() if m and b and "MMPROJ" not in m.upper()}
    if not methods:
        return ""
    if "Q4_K_M" in methods:
        return "Q4_K_M"
    return min(methods, key=lambda m: (abs(methods[m] - 4), m))


def referenced_files(text: str) -> set[str]:
    """Relative paths the card links to as images (markdown or HTML), e.g. './assets/logo.svg' -> 'assets/logo.svg'."""
    refs = re.findall(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)", text)
    refs += re.findall(r"""<(?:img|source)\b[^>]*?\b(?:src|srcset)\s*=\s*["']([^"'\s]+)""", text, re.IGNORECASE)
    out = set()
    for ref in refs:
        if re.match(r"^([a-z][a-z0-9+.-]*:|//|#|/)", ref, re.IGNORECASE):
            continue  # absolute URLs render anywhere
        out.add(unquote(ref.split("#", 1)[0].split("?", 1)[0]).removeprefix("./"))
    return out


def files_to_copy(siblings, text: str) -> list[str]:
    """License files, images and eval results from the source repo, plus anything the card links to."""
    refs = referenced_files(text)
    return sorted(
        s.rfilename for s in siblings
        if (s.size or 0) <= COPY_MAX_BYTES
        and not s.rfilename.startswith("original/")
        and (s.rfilename in refs
             or s.rfilename.startswith(".eval_results/")
             or any(fnmatch.fnmatch(PurePosixPath(s.rfilename).name, p) for p in COPY_PATTERNS))
    )


def publish_card(api, source: str, out_repo: str, files: list[tuple[str, int]], work: Path, enabled: bool) -> Path:
    """Commit README.md plus license/image files to HF in one commit, and keep the same files in `work`.

    `work` ends up mirroring the repo minus the GGUFs; the workflow saves it into this GitHub repo.
    """
    from huggingface_hub import CommitOperationAdd, ModelCard, hf_hub_download

    readme = build_card(api, source, out_repo, files)
    try:
        text = ModelCard.load(source).text
    except Exception:
        text = ""
    extras = files_to_copy(api.model_info(source, files_metadata=True).siblings or [], text)

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for rel in extras:
        hf_hub_download(source, rel, local_dir=work)
    shutil.rmtree(work / ".cache", ignore_errors=True)

    if enabled:
        ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode())]
        ops += [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(work / rel)) for rel in extras]
        api.create_commit(out_repo, operations=ops, commit_message="Update model card")
    else:
        print(f"[no-upload] would commit README.md and copy {extras}")

    # Written only after the HF commit succeeded, so the GitHub copy never differs from HF.
    path = work / "README.md"
    path.write_text(readme)
    print(f"Wrote {path} and {len(extras)} copied file(s)")
    return path

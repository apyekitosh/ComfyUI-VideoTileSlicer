"""Immutable artifacts plus one atomically replaced manifest. OS locks survive crashes."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import uuid
import numpy as np
import torch
from PIL import Image


@contextmanager
def locked(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".vts.lock").open("a+b") as handle:
        handle.seek(0)
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("This job is being updated by another execution; retry when it finishes") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read(root):
    p = Path(root)/"job.json"
    if not p.exists():
        return None
    state = json.loads(p.read_text("utf-8"))
    if state.get("schema") != 2:
        raise ValueError("Unsupported job format; use a new folder for VTS 2")
    return state


def publish(root, state):
    root = Path(root)
    p = root/(".manifest-"+uuid.uuid4().hex+".tmp")
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(p, root/"job.json")
    finally:
        p.unlink(missing_ok=True)


def path(root, relative):
    root = Path(root).resolve()
    candidate = (root/relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("Artifact path escapes job folder")
    return candidate


def remove(root, relative):
    p = path(root, relative)
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)


def save_checkpoint(root, images):
    relative = "checkpoints/"+uuid.uuid4().hex+".npy"
    p = path(root, relative)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        np.save(f, images.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    return relative


def load_checkpoint(root, relative):
    return torch.from_numpy(np.load(path(root, relative), allow_pickle=False).copy())


def save_png(p, frame):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = (frame.detach().cpu().clamp(0,1).numpy()*255).round().astype(np.uint8)
    with p.open("wb") as f:
        Image.fromarray(a).save(f, format="PNG")
        f.flush()
        os.fsync(f.fileno())


def load_png(p):
    with Image.open(p) as image:
        return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).float()/255


def link_or_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def raw_path(root, revision, local_index):
    return path(root, revision["raw"])/f"{local_index:06d}.png"


def raw_frames(root, revision, start, length):
    if start < 0 or start+length > revision["valid"]:
        raise ValueError("Requested context exceeds preserved predecessor frames")
    return torch.stack([load_png(raw_path(root, revision, i)) for i in range(start, start+length)])


def collect_orphans(root, state):
    """Recover abandoned transaction artifacts only; keep every recorded revision."""
    retained_checkpoints = {r["checkpoint"] for r in state["revisions"].values() if r["checkpoint"]}
    retained_raw = {str(Path(r["raw"]).parent).replace("\\", "/") for r in state["revisions"].values() if r["raw"]}
    candidates = []
    root = Path(root)
    for p in (root/"checkpoints").glob("*.npy"):
        if len(p.stem) == 32 and all(c in "0123456789abcdef" for c in p.stem):
            relative = p.relative_to(root).as_posix()
            if relative not in retained_checkpoints:
                candidates.append(relative)
    for p in (root/"sequence").glob("*"):
        if p.is_dir() and len(p.name) == 32 and all(c in "0123456789abcdef" for c in p.name):
            relative = p.relative_to(root).as_posix()
            if relative != state["sequence"]:
                candidates.append(relative)
    for p in (root/"segments").glob("*/revisions/*/*"):
        if p.is_dir() and len(p.name) == 32 and all(c in "0123456789abcdef" for c in p.name):
            relative = p.relative_to(root).as_posix()
            if relative not in retained_raw:
                candidates.append(relative)
    for relative in candidates:
        remove(root, relative)

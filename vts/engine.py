"""Transactional sequential tile engine, independent of ComfyUI and sampling."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid
import torch
from .geometry import Settings, frame_range
from . import imaging as im, storage as disk


@dataclass(frozen=True)
class Job:
    path: str
    source: torch.Tensor
    source_hash: str
    settings_json: str
    action: str = "continue"
    segment: int = 0
    request_id: int = 0
    revision: int = 1

    @property
    def settings(self):
        return Settings(**json.loads(self.settings_json)).validate()


@dataclass(frozen=True)
class Ticket:
    path: str
    job_id: str
    revision: str
    tile: int
    version: int
    nonce: str
    spec_json: str


def describe(state):
    if state is None:
        return {"status": "not started"}
    selected = []
    stale = False
    for index in sorted(map(int, state["selected"])):
        key = state["selected"][str(index)]
        r = state["revisions"][key]
        expected = state["selected"].get(str(index-1))
        stale = stale or r["predecessor"] != expected
        selected.append(dict(segment=index, revision=key, start=r["start"], end=r["start"]+r["valid"], stale_context=stale))
    active = state["revisions"].get(state["active"])
    return dict(job_id=state["id"], version=state["version"], active=active,
                selected=selected, sequence=state.get("sequence"), commands=state["commands"])


def _new_state(job):
    s = job.settings
    return dict(schema=2, id=uuid.uuid4().hex, source_hash=job.source_hash,
                total=len(job.source), width=s.width, height=s.height,
                version=0, active=None, selected={}, revisions={}, commands={}, cascade=[], sequence=None)


def _new_revision(state, index, start, valid, length, settings, prep, predecessor):
    number = 1 + max([r["number"] for r in state["revisions"].values() if r["index"] == index] or [0])
    key = f"{index:04d}:{number:04d}"
    prev = state["revisions"].get(predecessor)
    incoming = prev["start"]+prev["valid"]-start if prev else 0
    if incoming < 0 or incoming >= valid or incoming > length:
        raise ValueError("Temporal overlap must leave at least one new valid frame in the segment")
    if prev and start < prev["start"]+prev["incoming"]:
        raise ValueError("Temporal overlap would span three segments; reduce it")
    if settings.fade_frames > incoming and prev:
        raise ValueError("Fade length exceeds this segment's actual incoming overlap")
    r = dict(key=key, index=index, number=number, start=start, valid=valid, length=length,
             incoming=incoming, predecessor=predecessor, settings=settings.to_dict(), prep=prep,
             next_tile=0, checkpoint=None, raw=None, status="active", receipts={})
    state["revisions"][key] = r
    state["active"] = key
    return r


def _choose_revision(state, job, prep):
    s = job.settings
    command = str(job.request_id)
    signature = "restart_active" if job.action == "restart_active" else f"{job.action}:{job.segment}"
    if job.action not in ("continue", "revise", "regenerate_from", "restart_active"):
        raise ValueError("Unknown job action")
    if job.action == "restart_active" and command not in state["commands"]:
        if state["active"] is None:
            raise ValueError("No active segment to restart; use revise for a completed segment")
        old = state["revisions"][state["active"]]
        predecessor = state["selected"].get(str(old["index"]-1))
        if str(old["index"]) in state["selected"]:
            # A replacement of published work must not retime its neighbors.
            s = Settings(**{**s.to_dict(), "frames": old["length"],
                            "overlap_frames": old["incoming"],
                            "fade_frames": min(s.fade_frames, old["incoming"])}).validate()
            start, valid, length = old["start"], old["valid"], old["length"]
        else:
            prev = state["revisions"].get(predecessor)
            start = prev["start"]+prev["valid"]-s.overlap_frames if prev else 0
            if start < 0:
                raise ValueError("Temporal overlap exceeds predecessor length")
            valid, length = frame_range(start, state["total"], s)
        replacement = _new_revision(state, old["index"], start, valid, length, s, prep, predecessor)
        replacement["restarted_from"] = old["key"]
        old["status"] = "abandoned"
        old["checkpoint"] = None
        state["commands"][command] = signature
        # Preserve a pending regenerate_from cascade. All changes remain local
        # until Prepare successfully builds and publishes the new checkpoint.
        return replacement
    if job.action != "continue" and command not in state["commands"]:
        if state["active"]:
            raise ValueError("Use restart_active to replace unfinished work, or finish it before revising another segment")
        old_key = state["selected"].get(str(job.segment))
        if old_key is None:
            raise ValueError("Select a completed segment to revise")
        state["commands"][command] = signature
        old = state["revisions"][old_key]
        state["cascade"] = [i for i in sorted(map(int, state["selected"])) if i > job.segment] if job.action == "regenerate_from" else []
        # Revisions keep the historical range. Next-segment length changes only affect new segments.
        rsettings = Settings(**{**s.to_dict(), "frames": old["length"],
                                "overlap_frames": old["incoming"],
                                "fade_frames": min(s.fade_frames, old["incoming"])})
        rsettings.validate()
        return _new_revision(state, job.segment, old["start"], old["valid"], old["length"], rsettings,
                             prep, state["selected"].get(str(job.segment-1)))
    if job.action != "continue" and state["commands"].get(command) != signature:
        raise ValueError("Request ID was already used for a different action; increment request_id")
    if state["active"]:
        return state["revisions"][state["active"]]
    if state["cascade"]:
        index = state["cascade"].pop(0)
        old = state["revisions"][state["selected"][str(index)]]
        return _new_revision(state, index, old["start"], old["valid"], old["length"],
                             Settings(**old["settings"]), prep, state["selected"].get(str(index-1)))
    if job.action != "continue":
        return None  # Revision commands are one-shot; do not append unexpected segments.
    index = len(state["selected"])
    previous_key = state["selected"].get(str(index-1))
    previous = state["revisions"].get(previous_key)
    if previous and previous["start"]+previous["valid"] >= state["total"]:
        return None
    start = previous["start"]+previous["valid"]-s.overlap_frames if previous else 0
    if start < 0:
        raise ValueError("Temporal overlap exceeds predecessor length")
    valid, length = frame_range(start, state["total"], s)
    if previous and valid <= s.overlap_frames:
        raise ValueError("Trim policy cannot make forward progress; use pad for the final segment")
    return _new_revision(state, index, start, valid, length, s, prep, previous_key)


def _spec(revision, tile, version):
    return json.dumps(dict(revision=revision["key"], tile=tile, version=version,
                           start=revision["start"], valid=revision["valid"], length=revision["length"],
                           settings=revision["settings"], predecessor=revision["predecessor"], prep=revision["prep"]), sort_keys=True)


def prepare(job, control=None, references=None, user_mask=None, control_mode="workflow",
            reference_mode="tile", reference_transform="crop", sharpen=0.8, source_strength=0.1):
    im.validate_images(job.source, "source")
    if control_mode not in ("workflow", "source", "neutral", "supplied"):
        raise ValueError("Unknown control mode")
    if control_mode == "supplied" and control is None:
        raise ValueError("Connect a control video for supplied mode")
    if reference_mode not in ("single", "tile", "segment", "frame") or reference_transform not in ("crop", "fit", "stretch", "unchanged"):
        raise ValueError("Unknown reference mode or transform")
    if not 0 <= sharpen <= 1 or not 0 <= source_strength <= 1:
        raise ValueError("Sharpen and source strength must be in [0, 1]")
    for tensor, name in [(control, "control"), (references, "reference")]:
        if tensor is not None:
            im.validate_images(tensor, name)
    if user_mask is not None and (user_mask.ndim != 3 or not len(user_mask) or not torch.isfinite(user_mask).all()):
        raise ValueError("User mask must be finite [T,H,W]")
    prep = dict(control_mode=control_mode, reference_mode=reference_mode, reference_transform=reference_transform,
                sharpen=sharpen, source_strength=source_strength,
                control_hash=im.fingerprint(control) if control is not None else None,
                reference_hash=im.fingerprint(references) if references is not None else None,
                mask_hash=im.fingerprint(user_mask) if user_mask is not None else None)
    root = Path(job.path)
    with disk.locked(root):
        state = disk.read(root)
        if state is None:
            if any(p.name != ".vts.lock" for p in root.iterdir()):
                raise ValueError("New jobs require an empty folder; legacy jobs are not migrated automatically")
            state = _new_state(job)
            disk.publish(root, state)
        disk.collect_orphans(root, state)
        if state["source_hash"] != job.source_hash or state["total"] != len(job.source):
            raise ValueError("Source changed; use a new job folder")
        if (state["width"], state["height"]) != (job.settings.width, job.settings.height):
            raise ValueError("Output resolution is fixed for this job; use a new job folder")
        previous_checkpoint = state["revisions"][state["active"]]["checkpoint"] if state["active"] else None
        r = _choose_revision(state, job, prep)
        if r is None:
            return None
        if r["prep"] != prep:
            raise ValueError("Preparation inputs/settings changed during an active segment; use restart_active with a new request_id to apply them now, or restore the original inputs")
        s = Settings(**r["settings"])
        tile = s.tiles()[r["next_tile"]]
        source = im.sharpen(im.resize(im.select(job.source, r["start"], r["length"], len(job.source)), s.width, s.height), sharpen)
        source_tile = im.crop(source, tile)
        new_checkpoint = r["checkpoint"] is None
        if new_checkpoint:
            r["append_after"] = job.action == "continue"
            canvas = source.clone()
            if r["incoming"]:
                prev = state["revisions"][r["predecessor"]]
                canvas[:r["incoming"]] = disk.raw_frames(root, prev, r["start"]-prev["start"], r["incoming"])
            state["version"] += 1
        else:
            canvas = disk.load_checkpoint(root, r["checkpoint"])
        context = im.crop(canvas, tile)
        mask, blend = im.masks(tile, s, r["length"], r["incoming"])
        if user_mask is not None:
            custom = im.crop(im.resize(im.select(user_mask, r["start"], r["length"], len(job.source), True), s.width, s.height, True), tile).clamp(0,1)
            mask *= custom
        if control_mode == "workflow":
            rgb = source_tile*source_strength + (1-source_strength)
            base = (rgb*torch.tensor([0.2126,0.7152,0.0722])).sum(-1, keepdim=True).expand_as(rgb)
        elif control_mode == "source":
            base = source_tile
        elif control_mode == "neutral":
            base = torch.full_like(source_tile, 0.5)
        else:
            base = im.crop(im.resize(im.select(control, r["start"], r["length"], len(job.source), True), s.width, s.height), tile)
        prepared_control = base*mask[...,None]+context*(1-mask[...,None])
        ref = im.reference(references, reference_mode, reference_transform, r, r["next_tile"], tile, s, source)
        if new_checkpoint:
            checkpoint = disk.save_checkpoint(root, canvas)
            r["checkpoint"] = checkpoint
            try:
                disk.publish(root, state)
            except BaseException:
                disk.remove(root, checkpoint)
                raise
            if previous_checkpoint:
                # Only retire old partial pixels after the restart is durable.
                # Orphan collection retries cleanup if a reader still holds it.
                try:
                    disk.remove(root, previous_checkpoint)
                except OSError:
                    pass
        spec = _spec(r, r["next_tile"], state["version"])
        nonce = hashlib.sha256((state["id"]+spec).encode()).hexdigest()
        ticket = Ticket(str(root.resolve()), state["id"], r["key"], r["next_tile"], state["version"], nonce, spec)
        return dict(source=source_tile, control=prepared_control, mask=mask, reference=ref,
                    blend=blend, context=context, ticket=ticket,
                    info={**describe(state), "tile": tile, "pending_settings": job.settings.to_dict() != r["settings"]})


def _save_raw(root, r, canvas):
    relative = f"segments/{r['index']:04d}/revisions/{r['number']:04d}/{uuid.uuid4().hex}"
    directory = disk.path(root, relative)
    try:
        for i, frame in enumerate(canvas[:r["valid"]]):
            disk.save_png(directory/"raw"/f"{i:06d}.png", frame)
        # Indexed boundary folders are convenience views; raw remains authoritative.
        for label, indices in [("leading", range(r["incoming"])),
                               ("trailing", range(max(0, r["valid"]-r["settings"]["overlap_frames"]), r["valid"]))]:
            for i in indices:
                disk.link_or_copy(directory/"raw"/f"{i:06d}.png", directory/label/f"{r['start']+i:06d}.png")
        return relative+"/raw", relative
    except BaseException:
        if directory.exists():
            disk.remove(root, relative)
        raise


def _export(root, state):
    relative = "sequence/"+uuid.uuid4().hex
    directory = disk.path(root, relative)
    directory.mkdir(parents=True)
    try:
        for index in sorted(map(int, state["selected"])):
            r = state["revisions"][state["selected"][str(index)]]
            prev_key = state["selected"].get(str(index-1))
            prev = state["revisions"].get(prev_key)
            incoming = prev["start"]+prev["valid"]-r["start"] if prev else 0
            fade = min(r["settings"]["fade_frames"], incoming)
            weights = torch.cat([torch.zeros(incoming-fade), im.ramp(fade, r["settings"]["curve"])])
            for i in range(r["valid"]):
                target = directory/f"image-{r['start']+i:06d}.png"
                if i < incoming:
                    a = disk.load_png(disk.raw_path(root, prev, r["start"]+i-prev["start"]))
                    b = disk.load_png(disk.raw_path(root, r, i))
                    # Unlink before writing: target may be hard-linked to predecessor raw.
                    target.unlink(missing_ok=True)
                    disk.save_png(target, a*(1-weights[i])+b*weights[i])
                else:
                    disk.link_or_copy(disk.raw_path(root, r, i), target)
        return relative
    except BaseException:
        disk.remove(root, relative)
        raise


def commit(ticket, generated):
    im.validate_images(generated, "generated tile")
    root = Path(ticket.path)
    with disk.locked(root):
        state = disk.read(root)
        if state is None or state["id"] != ticket.job_id:
            raise ValueError("Ticket belongs to a different job")
        r = state["revisions"].get(ticket.revision)
        if r is None:
            raise ValueError("Unknown revision ticket")
        if ticket.nonce in r["receipts"]:
            return dict(duplicate=True, segment_complete=r["status"] == "complete", continue_run=False,
                        info=describe(state), images=None, sequence=state["sequence"])
        spec = _spec(r, ticket.tile, state["version"])
        if (state["active"] != ticket.revision or state["version"] != ticket.version or
                r["next_tile"] != ticket.tile or spec != ticket.spec_json or
                hashlib.sha256((state["id"]+spec).encode()).hexdigest() != ticket.nonce):
            raise ValueError("Stale or mismatched ticket: prepare the current tile again")
        s = Settings(**r["settings"])
        tile = s.tiles()[ticket.tile]
        if tuple(generated.shape) != (r["length"], tile["height"], tile["width"], 3):
            raise ValueError(f"Generated shape {tuple(generated.shape)} does not match {(r['length'],tile['height'],tile['width'],3)}")
        canvas = disk.load_checkpoint(root, r["checkpoint"])
        _, blend = im.masks(tile, s, r["length"], r["incoming"])
        x,y,w,h = tile["x"],tile["y"],tile["valid_width"],tile["valid_height"]
        weight = blend[:, :h, :w, None]
        canvas[:, y:y+h, x:x+w] = canvas[:, y:y+h, x:x+w]*(1-weight)+im.cpu(generated)[:, :h, :w]*weight
        old_checkpoint, old_sequence = r["checkpoint"], state["sequence"]
        artifacts = []
        try:
            r["next_tile"] += 1
            r["receipts"][ticket.nonce] = ticket.tile
            complete = r["next_tile"] == len(s.tiles())
            if complete:
                r["raw"], raw_dir = _save_raw(root, r, canvas)
                artifacts.append(raw_dir)
                r["checkpoint"] = None
                r["status"] = "complete"
                state["selected"][str(r["index"])] = r["key"]
                state["active"] = None
                state["sequence"] = _export(root, state)
                artifacts.append(state["sequence"])
            else:
                r["checkpoint"] = disk.save_checkpoint(root, canvas)
                artifacts.append(r["checkpoint"])
            state["version"] += 1
            disk.publish(root, state)
        except BaseException:
            for artifact in artifacts:
                disk.remove(root, artifact)
            raise
        # Cleanup after publication cannot invalidate a successful commit.
        for artifact in [old_checkpoint, old_sequence if complete else None]:
            if artifact:
                try:
                    disk.remove(root, artifact)
                except OSError:
                    pass
        more = not complete or bool(state["cascade"]) or (r.get("append_after", False) and r["start"]+r["valid"] < state["total"])
        return dict(duplicate=False, segment_complete=complete, continue_run=more,
                    images=canvas, info=describe(state), sequence=state["sequence"])


def restore_revision(root, segment, revision, request_id):
    """Explicit one-shot rollback; no rendering. Never modifies preserved revisions."""
    with disk.locked(root):
        state = disk.read(root)
        if state is None or state["active"]:
            raise ValueError("Restore requires an existing job with no active segment")
        key = f"{segment:04d}:{revision:04d}"
        signature = f"restore:{key}"
        command = str(request_id)
        if command in state["commands"]:
            if state["commands"][command] != signature:
                raise ValueError("Increment request_id for a new action")
            return describe(state)
        r = state["revisions"].get(key)
        if r is None or r["status"] != "complete":
            raise ValueError("Revision is not complete")
        old_sequence = state["sequence"]
        state["selected"][str(segment)] = key
        state["commands"][command] = signature
        state["sequence"] = _export(root, state)
        state["version"] += 1
        try:
            disk.publish(root, state)
        except BaseException:
            disk.remove(root, state["sequence"])
            raise
        if old_sequence:
            try:
                disk.remove(root, old_sequence)
            except OSError:
                pass
        return describe(state)


def preview(root, mode="sequence", segment=0, revision=0, start=0, count=33):
    """Read-only, bounded frame loading. No progress changes or source needed."""
    with disk.locked(root):
        state = disk.read(root)
        if state is None:
            raise ValueError("Job has not started")
        if mode == "active":
            r = state["revisions"].get(state["active"])
            if r is None:
                raise ValueError("No active segment; choose sequence or revision")
            import numpy as np
            array = np.load(disk.path(root, r["checkpoint"]), mmap_mode="r", allow_pickle=False)
            frames = torch.from_numpy(array[start:start+count].copy())
        else:
            if mode == "sequence":
                if not state["sequence"]:
                    raise ValueError("No completed segments yet")
                directory = disk.path(root, state["sequence"])
            else:
                key = f"{segment:04d}:{revision:04d}" if revision else state["selected"].get(str(segment))
                r = state["revisions"].get(key)
                if not r or not r["raw"]:
                    raise ValueError("No completed revision at this selection")
                directory = disk.path(root, r["raw"])
                if mode in ("leading", "trailing"):
                    directory = directory.parent/mode
            files = sorted(directory.glob("*.png"))[start:start+count]
            if not files:
                raise ValueError("No frames at this preview range")
            frames = torch.stack([disk.load_png(p) for p in files])
        if not len(frames):
            raise ValueError("No frames at this preview range")
        return frames, describe(state)

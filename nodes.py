"""Thin ComfyUI V3 adapters. No model, conditioning, VAE or sampler dependencies."""
import json
from pathlib import Path
from comfy_api.latest import io, ui
import folder_paths
from .vts.geometry import Settings
from .vts import engine, imaging

JOB = io.Custom("VTS2_JOB")
TICKET = io.Custom("VTS2_TICKET")


def job_path(value):
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(folder_paths.get_output_directory())/p
    return str(p.resolve())


def text(value):
    return json.dumps(value, indent=2)


def preview_ui(images, fps, cls, enabled=True):
    if not enabled or images is None:
        return {}
    images = imaging.cpu(images[:121])
    scale = min(1, 768/max(images.shape[1:3]))
    if scale < 1:
        images = imaging.resize(images, max(1,round(images.shape[2]*scale)), max(1,round(images.shape[1]*scale)))
    saved = ui.ImageSaveHelper.save_animated_webp(images, "vts2_preview", io.FolderType.temp,
                                                cls=cls, fps=float(fps), lossless=False, quality=80, method=2)
    return ui.PreviewVideo(saved).as_dict()


class TileJob(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="VTS2_Job", display_name="VTS Tile Job", category="VTS",
            description='Defines the source, output size, tile layout and temporal segments for a saved job. Connect it to Prepare Tile. Active segments keep their settings until finished or explicitly restarted. Use restart_active to retry unfinished work with new settings, or revise and restore for completed segments.',
            inputs=[io.Image.Input("source", tooltip='Full source video as an RGB image batch. Frame order defines the job timeline.'), io.String.Input("job_folder", default="vts/my_job", tooltip='Folder for checkpoints, revisions and the sequence. Relative paths use ComfyUI output; new jobs need an empty folder.'),
                    io.Int.Input("width", default=1920, min=1, tooltip='Final full-frame width in pixels. Fixed for this job.'), io.Int.Input("height", default=1088, min=1, tooltip='Final full-frame height in pixels. Fixed for this job.'),
                    io.Int.Input("tile_width", default=1024, min=1, tooltip='Width of each tile sent to your sampling chain. Edge padding is removed when stitching.'), io.Int.Input("tile_height", default=624, min=1, tooltip='Height of each tile sent to your sampling chain. Edge padding is removed when stitching.'),
                    io.Int.Input("frames", default=33, min=1, tooltip='Length of the next new segment, including overlap. restart_active can change it for unpublished work; published segment ranges stay fixed.'),
                    io.Int.Input("overlap_frames", default=9, min=0, tooltip='Frames shared with the preceding segment. Must leave room for new frames; independent of fade length.'), io.Int.Input("fade_frames", default=9, min=0, tooltip='Length of the crossfade at the end of the shared frame range. Cannot exceed temporal overlap.'),
                    io.Int.Input("overlap_x", default=32, min=0, advanced=True, tooltip='Minimum horizontal tile overlap for minimum_overlap layout. Ignored by workflow layout; separate from feather width.'), io.Int.Input("overlap_y", default=32, min=0, advanced=True, tooltip='Minimum vertical tile overlap for minimum_overlap layout. Ignored by workflow layout; separate from feather height.'),
                    io.Int.Input("feather_x", default=32, min=0, tooltip='Width of the protected left border and its stitching ramp where a tile meets an earlier column.'), io.Int.Input("feather_y", default=32, min=0, tooltip='Height of the protected top border and its stitching ramp where a tile meets an earlier row.'),
                    io.Combo.Input("layout", options=["workflow", "minimum_overlap"], default="workflow", tooltip='workflow spreads the smallest tile grid across the frame. minimum_overlap adds tiles as needed to meet overlap_x/y.'),
                    io.Combo.Input("alignment", options=["wan", "none"], default="wan", tooltip='wan requires 4n+1 frames and tile dimensions divisible by 16. none leaves alignment to your sampling chain.'),
                    io.Combo.Input("end_policy", options=["pad", "trim"], default="pad", tooltip='pad repeats the last source frame, then removes extra frames from export. trim shortens the final segment to a valid length.'),
                    io.Combo.Input("curve", options=["linear", "smoothstep"], default="linear", tooltip='Shape of spatial and temporal fades: linear is uniform; smoothstep eases in and out.'),
                    io.Float.Input("fps", default=24, min=0.01, max=240, tooltip='Playback rate saved with the segment for previews. Does not resample or change the source frame count.'),
                    io.Combo.Input("action", options=["continue", "revise", "regenerate_from", "restore", "restart_active"], default="continue", tooltip='restart_active retries unfinished work from tile 0 with current settings. continue resumes/appends; revise replaces completed work; regenerate_from also rerenders later segments; restore selects a saved revision.'),
                    io.Int.Input("segment", default=0, min=0, tooltip='Zero-based completed segment to revise, regenerate from or restore. Ignored for continue and restart_active.'), io.Int.Input("revision", default=1, min=1, advanced=True, tooltip='Saved revision number to select with restore, starting at 1. Ignored for other actions.'),
                    io.Int.Input("request_id", default=0, min=0, tooltip='Increment for each new restart, revision or restore command. Keep fixed while that command processes its tiles.')],
            outputs=[JOB.Output("job", tooltip='Source and requested job settings. Connect to Prepare Tile to resume or start work.')])

    @classmethod
    def execute(cls, source, job_folder, action="continue", segment=0, revision=1, request_id=0, **settings):
        imaging.validate_images(source, "source")
        s = Settings(**settings).validate()
        return io.NodeOutput(engine.Job(job_path(job_folder), source, imaging.fingerprint(source),
                                        json.dumps(s.to_dict(), sort_keys=True), action, segment, request_id, revision))


class PrepareTile(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="VTS2_Prepare", display_name="VTS Prepare Tile", category="VTS", not_idempotent=True,
            description='Prepares the next tile with source frames, control video, masks and a reference, using previously rendered neighbors for continuity. Connect these outputs to your own conditioning and sampling chain, and pass the ticket directly to Commit Tile. Progress advances only when the generated tile is committed.',
            inputs=[JOB.Input("job", tooltip='Job definition from Tile Job, including its source and saved progress location.'), io.Combo.Input("control_mode", options=["workflow", "source", "neutral", "supplied"], default="workflow", tooltip='Control for regions to generate: workflow uses whitened grayscale source; source uses RGB; neutral uses gray; supplied uses the control input.'),
                    io.Combo.Input("reference_mode", options=["single", "tile", "segment", "frame"], default="tile", tooltip='Choose the first reference, one per tile, one per segment, or the reference at the segment start frame. Missing entries raise an error.'),
                    io.Combo.Input("reference_transform", options=["crop", "fit", "stretch", "unchanged"], default="crop", tooltip='crop uses the tile location in the full frame; fit center-crops to fill the tile; stretch resizes to it; unchanged preserves the reference.'),
                    io.Float.Input("sharpen", default=0.8, min=0, max=1, step=0.05, tooltip='Contrast-adaptive sharpening applied to the resized source before cropping. Set 0 to disable.'),
                    io.Float.Input("source_strength", default=0.1, min=0, max=1, step=0.05, tooltip='Source contribution before grayscale conversion in workflow control mode. 0 gives white; 1 keeps source luminance.'),
                    io.Combo.Input("preview", options=["off", "source", "control", "mask", "reference", "blend", "context"], default="control", tooltip='Input or mask to display on this node. context shows the current canvas crop; off skips preview encoding.'),
                    io.Image.Input("control", optional=True, tooltip='Optional RGB control sequence for supplied mode. Use the full source frame count, or one frame to repeat.'), io.Image.Input("references", optional=True, tooltip="Optional reference image or batch, selected by reference_mode. If absent, uses the segment's first source frame."), io.Mask.Input("user_mask", optional=True, tooltip='Optional mask multiplied into the control mask: 0 preserves context, 1 allows generation. Use one frame or the full source frame count.')],
            outputs=[io.Image.Output("source_tile", tooltip='Resized, sharpened source crop for this tile and segment. Use for external VAE encoding or other source-based processing.'), io.Image.Output("control_video", tooltip='Prepared control frames with rendered spatial and temporal context inserted into protected areas.'), io.Mask.Output("control_mask", tooltip='Conditioning mask: 0 preserves context, 1 allows generation. No latent noise mask is attached automatically.'),
                     io.Image.Output("reference", tooltip='Reference selected and transformed for the current tile, ready for your conditioning node.'), io.Mask.Output("blend_mask", tooltip='Weight of the new tile during stitching: 0 keeps the canvas, 1 uses the result. Commit calculates this separately from the control mask.'), TICKET.Output("ticket", tooltip='Identity and frozen settings for this exact tile. Connect directly to the Commit node receiving its generated images.'),
                     io.Int.Output("width", tooltip='Prepared tile width in pixels, including any edge padding.'), io.Int.Output("height", tooltip='Prepared tile height in pixels, including any edge padding.'), io.Int.Output("frames", tooltip='Prepared frame count, including any final-segment padding. Use as the conditioning or sampling length.'), io.String.Output("info", tooltip='JSON summary of the active segment, tile coordinates, saved progress and pending settings.')])

    @classmethod
    def execute(cls, job, preview="control", **kwargs):
        if job.action == "restore":
            result = engine.restore_revision(job.path, job.segment, job.revision, job.request_id)
            from comfy_execution.graph import ExecutionBlocker
            return io.NodeOutput(*[ExecutionBlocker(None) for _ in range(10)], ui={"text": [text(result)]})
        result = engine.prepare(job, **kwargs)
        if result is None:
            from comfy_execution.graph import ExecutionBlocker
            return io.NodeOutput(*[ExecutionBlocker(None) for _ in range(10)], ui={"text": ["Requested work is complete."]})
        shown = result.get(preview)
        if shown is not None and shown.ndim == 3:
            shown = shown[..., None].expand(-1,-1,-1,3)
        fps = result["info"]["active"]["settings"]["fps"]
        u = preview_ui(shown, fps, cls, preview != "off")
        u["text"] = [text(result["info"])]
        source = result["source"]
        return io.NodeOutput(source, result["control"], result["mask"], result["reference"], result["blend"],
                             result["ticket"], source.shape[2], source.shape[1], len(source), text(result["info"]), ui=u)


class CommitTile(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="VTS2_Commit", display_name="VTS Commit Tile", category="VTS", is_output_node=True,
            not_idempotent=True, inputs=[io.Image.Input("generated", tooltip="Decoded RGB result for this tile. Width, height and frame count must match Prepare Tile's outputs."), TICKET.Input("ticket", tooltip='Ticket from the same Prepare Tile execution that produced the sampling inputs. Stale or mismatched results are rejected.'),
                io.Combo.Input("preview", options=["active", "tile", "off"], default="active", tooltip='Show the assembled segment canvas, the incoming tile, or no preview. Saving is unaffected.'),
                io.Combo.Input("auto_queue", options=["off", "until_segment_complete", "continuous"], default="off", tooltip='off runs one tile; until_segment_complete pauses between segments; continuous keeps going. Requires the browser extension and a successful graph run.')],
            outputs=[io.Image.Output("canvas", tooltip='Full segment canvas after this tile was blended in. May still contain unprocessed regions and final-segment padding.'), io.String.Output("sequence_folder", tooltip='Current published PNG sequence folder, or empty before the first segment finishes. Changes when a new sequence is published.'), io.String.Output("info", tooltip='JSON summary of saved progress, selected revisions and any outdated downstream context.')], description='Blends one generated tile into the segment canvas and saves a resumable checkpoint. When the segment is complete, it publishes a new image sequence while preserving earlier revisions. Connect the ticket from the same Prepare Tile execution; optional autoqueue continues after the whole graph succeeds.')

    @classmethod
    def execute(cls, generated, ticket, preview="active", auto_queue="off"):
        result = engine.commit(ticket, generated)
        s = json.loads(ticket.spec_json)["settings"]
        images = result["images"]
        u = preview_ui(generated if preview == "tile" else images, s["fps"], cls, preview != "off")
        u["text"] = [text(result["info"])]
        u["vts2"] = [dict(nonce=ticket.nonce, job=ticket.job_id,
                           continue_run=result["continue_run"] and not result["duplicate"] and
                           (auto_queue == "continuous" or (auto_queue == "until_segment_complete" and not result["segment_complete"])))]
        if images is None:
            from comfy_execution.graph import ExecutionBlocker
            images = ExecutionBlocker(None)
        seq = str(Path(ticket.path)/result["sequence"]) if result["sequence"] else ""
        return io.NodeOutput(images, seq, text(result["info"]), ui=u)


class JobPreview(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="VTS2_Preview", display_name="VTS Job Preview", category="VTS", is_output_node=True,
            not_idempotent=True,
            inputs=[io.String.Input("job_folder", default="vts/my_job", tooltip="Existing saved job to inspect. Relative paths use ComfyUI's output folder."),
                    io.Combo.Input("mode", options=["sequence", "active", "revision", "leading", "trailing"], default="sequence", tooltip='sequence shows the published result; active shows unfinished work; revision shows pre-crossfade frames; leading/trailing show saved overlap frames.'),
                    io.Int.Input("segment", default=0, min=0, tooltip='Zero-based segment to inspect in revision, leading or trailing mode.'), io.Int.Input("revision", default=0, min=0, tooltip="Historical revision number, starting at 1. Set 0 to use the segment's currently selected revision."),
                    io.Int.Input("start", default=0, min=0, tooltip='Zero-based position of the first frame within the selected view. For boundary views, this is relative to that boundary folder.'), io.Int.Input("count", default=33, min=1, max=121, tooltip='Maximum number of frames to load and preview, up to 121. Fewer are returned at the end of the view.')],
            outputs=[io.Image.Output("images", tooltip='Selected frames at their stored resolution. Only the on-node playback preview is downscaled.'), io.String.Output("info", tooltip='JSON summary of the job, active work, selected revisions and outdated context dependencies.')], description='Inspects a saved job without running sampling or changing its progress. View the current sequence, unfinished canvas, a completed revision or its boundary frames. Select a short frame range for playback; returned images retain their stored resolution.')

    @classmethod
    def execute(cls, job_folder, mode="sequence", segment=0, revision=0, start=0, count=33):
        root = job_path(job_folder)
        if not (Path(root)/"job.json").exists():
            raise ValueError("No VTS 2 job found at this path")
        images, info = engine.preview(root, mode, segment, revision, start, count)
        from .vts import storage
        state = storage.read(root)
        key = state["active"] or state["selected"].get(str(segment)) or list(state["selected"].values())[-1]
        fps = state["revisions"][key]["settings"]["fps"]
        u = preview_ui(images, fps, cls)
        u["text"] = [text(info)]
        return io.NodeOutput(images, text(info), ui=u)


NODES = [TileJob, PrepareTile, CommitTile, JobPreview]

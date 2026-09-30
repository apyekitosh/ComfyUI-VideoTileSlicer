# VideoTileSlicer 2

Four ComfyUI V3 nodes for preparing video tiles, assembling generated results,
checkpointing progress, and revisiting completed segments. Conditioning, VAE
encoding/decoding, models and sampling stay in your graph.

## Quick start

Restart ComfyUI after installing this folder and refresh the browser. Nodes are
under **VTS**. Open `examples/pass_through.json` for a small, model-free test.
Its Commit node queues until the first segment is complete; queue again for the
second segment. Use a new job folder for another independent test.

For a real workflow:

1. Connect your video loader's IMAGE batch to **VTS Tile Job / source**.
2. Connect Job to **VTS Prepare Tile**.
3. Route `source_tile` to your VAE encoder, and `control_video`, `control_mask`
   and `reference` to your chosen conditioning nodes. Connect the width, height
   and frame outputs if those nodes require them.
4. Decode the sampler result to IMAGE and connect it to **VTS Commit Tile**.
5. Connect Prepare's `ticket` directly to that same Commit node.

`control_mask` means **0 = preserve context, 1 = generate**. It is a conditioning
mask, not an automatically attached latent noise mask. `blend_mask` is a separate
stitching weight; Commit recomputes it from the frozen ticket settings.
User masks affect control preparation only; Commit does not apply them again.

`job_folder` can be absolute or relative to ComfyUI's output folder. Use an empty
folder for a new job. V1 job folders and V1 graphs are not automatically migrated.

## Variable segment lengths

The `frames` widget is the length for the **next new segment**. All tiles in an
active segment retain the same frozen frame range, geometry, fade and preparation
settings. Changing geometry/length widgets while it runs takes effect on the next
segment and is reported as pending in the info output. Preparation images and
preprocessing options must remain unchanged until the segment completes, unless
you explicitly restart it with the new settings.

### Restart an unfinished segment

If the first tile looks wrong, pause autoqueue (or interrupt the current run),
adjust your settings, choose **action = restart_active**, increment **request_id**,
and queue again. The active segment starts over at tile 0. You do not need to
finish the remaining tiles or change the segment selector.

This discards the unfinished attempt's tile pixels after the replacement is
safely saved; completed segments and the published sequence remain intact. The
new attempt uses current preparation inputs and tile settings. For a segment
that has never been published, you can also change its length and temporal
overlap. When retrying a revision of published work, its frame range stays fixed
so neighboring segments are not retimed. Source and full-frame output size
remain fixed for the job.

Keep request_id unchanged while the restarted attempt runs; otherwise each new
ID requests another restart. After it completes, switch to **continue** to append
more segments. A pending regenerate_from cascade is preserved and completes
before stopping. If the new settings are invalid or saving fails, the old active
attempt remains resumable.

For example, with nine overlap frames:

| Segment | Requested length | Global range (inclusive) |
|---|---:|---|
| 0 | 81 | 0–80 |
| 1 | 53 | 72–124 |
| 2 | 121 | 116–236 |

Choose `auto_queue = until_segment_complete` to pause at those boundaries and
change the next length. `continuous` processes the whole source with the current
settings. `off` processes one tile per manual queue. The Pause button stops
continuation after the current tile; it does not discard the committed result.

Temporal overlap may also change between segments, provided it fits inside the
predecessor's preserved frames and leaves at least one new frame. Triple temporal
overlaps are rejected. Temporal fade is independent of overlap: when shorter,
the earlier part of the overlap holds the predecessor, then the fade runs over
the last `fade_frames`. A one-frame fade chooses the new frame.

`alignment = wan` requires tile dimensions divisible by 16 and lengths of `4n+1`.
`none` imposes no model alignment. **pad** repeats the source's final frame to
fill the requested segment, and never exports padded frames. **trim** shortens
to the largest available aligned length; if that cannot advance beyond the
overlap, it asks you to use pad. Pad is the recommended final-segment policy.

## Spatial layout and controls

- **workflow:** uses the minimum grid based on tile dimensions and spreads tiles
  across the canvas. The saved workflow's 1920×1088 / 1024×624 setup produces
  four tiles, with 128 horizontal and 160 vertical overlap pixels.
- **minimum_overlap:** also uses the requested `overlap_x/y` when choosing how
  many tiles to create. Actual overlap may be larger to meet the far edge.
- **feather_x/y:** independent protected border and blending-ramp widths. The
  default 32-pixel feather does not expand to the full coverage overlap.

Tiles larger than the canvas are edge-padded for conditioning/sampling, then
cropped back during Commit. Output resolution and source identity are fixed for
the job. Tile dimensions may change for a new segment or a new revision.

Controls:

- **workflow:** sharpen source, mix it toward white (default source contribution
  0.1), desaturate with Rec.709, then insert spatial/temporal rendered context.
- **source:** use the source RGB as the generated-region control.
- **neutral:** use 0.5 gray there.
- **supplied:** use an externally prepared RGB control sequence there.

Optional control sequences and masks must have the source frame count, or one
frame for broadcast. Zero-mask areas use the current context canvas. There is no
silent clamping of mismatched sequence lengths. Source/control files are not
modified. Source tensors should be RGB floats in [0,1]. Sharpen=0 disables CAS;
positive strengths use the source workflow's CAS algorithm.

References have explicit selection modes: single, per tile, per segment, or the
source frame at the segment start. An undersized reference batch is an error,
not a silent repeat of the final image. Transform can crop by the tile rectangle,
fit by center-cropping, stretch, or remain unchanged. Without a supplied reference,
the current segment's first source frame is cropped to the tile.

## Revisions and rollback

Set Job's `action`, zero-based `segment`, and a new `request_id`:

- **revise:** render a new revision of that completed segment.
- **regenerate_from:** revise it, then regenerate each already-completed later
  segment using its stored range and geometry. Afterward switch to continue to
  append further segments if the original job was incomplete.
- **restore:** select a completed historical `revision` without sampling.
- **continue:** resume the active segment, or append the next new segment.
- **restart_active:** restart unfinished work from tile 0 with current settings;
  ignores the segment selector.

Keep request_id fixed throughout a command's tile runs. Increment it for the next
command, including restart and rollback. Revising a different completed segment
or restoring one requires no active unfinished segment; use restart_active to
retry the current attempt instead. Historical ranges stay fixed; changing `frames` does not retime existing
segments. A partial replacement is never published as the selected revision.

Replacing a predecessor rebuilds both adjacent sequence boundaries from raw
frames. Later generated content is retained and marked `stale_context` until
regenerated. Restoring the original predecessor clears that flag when the
recorded dependency chain matches again. Reblending cannot change the context
used by an earlier sampling execution.

The raw leading frames already include the workflow's per-tile temporal
compositing; "raw" here means **before the final sequence crossfade**. As in the
original workflow, the final crossfade is a separate operation.

## Storage and crash recovery

```
job.json                         # authoritative atomically replaced manifest
checkpoints/<id>.npy              # one current float32 assembled canvas
segments/0000/revisions/0001/<id>/
    raw/                         # completed pre-sequence-blend PNG frames
    leading/                     # indexed raw incoming boundary view
    trailing/                    # indexed raw outgoing boundary view
sequence/<generation>/           # current derived PNG sequence
```

Commit returns the current `sequence_folder`. That generation path changes when
a segment completes or a revision is selected; `job.json` always points to the
current one. Old exported generations are removed after successful publication.
Completed raw revisions remain available for rollback. Boundary views and
unblended sequence frames use hard links on supporting filesystems, falling back
to copies otherwise. Treat files within the job as managed, immutable artifacts;
copy an export elsewhere before editing it externally.

No individual generated tiles are retained. A commit stages a new canvas (or
completed revision and sequence), then atomically replaces the manifest. Progress
advances only after this succeeds. Previous artifacts remain valid on failure;
unpublished transaction artifacts are recovered on the next Prepare. An OS lock
prevents overlapping writers, and tickets reject stale results. Duplicate
commits do not blend twice or autoqueue. Failed sampling leaves the same tile
pending. Use local storage for the strongest filesystem locking/rename behavior.

Working checkpoints use float32 so restarting between tiles does not add PNG
quantization. Completed frames use RGB 8-bit PNG, matching the original sequence
format. One active float32 canvas costs roughly `frames × width × height × 12`
bytes, with temporary space for a second checkpoint during commit. CPU RAM must
also hold the source/prepared segment and assembly tensors. Full source loading
is currently delegated to your video loader; streaming video decoding is not
implemented. No GPU/model is loaded by these nodes.

Preview encoding failures do not undo a committed result: retrying its ticket is
safe. Autoqueue waits for the entire prompt to succeed, stays scoped to the
executing client's graph, and pauses on graph edits, failures, or other queued
work. API/headless callers queue explicitly; browser autoqueue is optional.

## Preview and export

Prepare can display source/control/mask/reference/blend/context. Commit shows the
tile or current canvas. Job Preview opens a saved sequence, active canvas,
selected/historical raw revision, or indexed boundary frames without generation.
Preview reads a bounded range (at most 121 frames) and encodes at most 768 pixels
on the long side. Previews use the stored FPS and do not advance progress.

The permanent output is a numbered PNG sequence. MP4 encoding/audio muxing stay
with existing video nodes or an external encoder; source audio is not processed.

## Development

```
python -m pytest tests -q
node tests/test_autoqueue.mjs
python tests/check_comfy_api.py --comfy-root /path/to/ComfyUI
```

The pure core (`vts/`) has no ComfyUI/model imports. Tests cover geometry, mask
semantics, variable segment lengths, checkpoint recovery, idempotent commits,
revisions, rollback, source validation and autoqueue event handling. The optional
API check exercises real V3 schemas and animated preview output on CPU. An
optional CAS parity test uses the locally installed essentials node source.

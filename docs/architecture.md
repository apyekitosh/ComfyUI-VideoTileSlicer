# State and transaction contract

`Settings` is immutable. `Job` describes requested next-segment settings and the
source fingerprint. `Ticket` is an immutable identity for one prepared tile,
including a canonical serialized specification and the manifest version it read.
The source is hashed at Tile Job execution; normal ComfyUI input caching reuses
that result between unchanged tile runs. Prepare is deliberately not cached.

Prepare can create an active revision/checkpoint, but never advances next_tile.
An explicit restart_active command creates a new attempt at tile zero and marks
the unfinished attempt abandoned. Its checkpoint is retired only after the new
checkpoint and manifest are published. Published selections and any regeneration
cascade stay intact; unpublished segment ranges can be recalculated from current
settings. The command ID is consumed once, so requeuing resumes the new attempt.
Commit is the sole owner of rendered progress. Each committed ticket has a
receipt. A receipt turns a repeated commit into a no-op. An unrecognized ticket
with a different version/specification is rejected.

The manifest is the commit point. Referenced artifacts are immutable until the
manifest no longer needs them. Working checkpoints retain exact float32 pixels;
completed revisions retain pre-sequence-crossfade PNGs. The derived sequence is
constructed from the selected revision of each segment, not the previous
derived sequence. Replacing a boundary never compounds an old crossfade.

Selection and dependency are separate: a revision records which predecessor it
used; selection says which revisions form the current export. A mismatch marks
downstream context stale, transitively. Historical revision data is preserved.

Frame intervals are half-open internally. Sequential segments satisfy
`next.start = previous.start + previous.valid - next.incoming`. The final valid
count is distinct from padded tensor length. A segment must advance the covered
range; triple temporal overlaps are rejected. Spatial tiles are sequential
because control preparation consumes already rendered neighbor pixels.

Scope deliberately excludes latent noise schedules, VACE model patching, VAE
operations, seeds, sampling, prompt selection, audio, and video encoding. Native
IMAGE/MASK outputs leave those decisions with the external graph.

Known first-version limits: entire source IMAGE batch is supplied by the loader;
one active segment per job; no arbitrary committed-tile rollback; no pruning UI
for historical revisions; 8-bit completed PNG storage; no cross-machine queue
coordinator. The repository's previous implementation is backed up separately.

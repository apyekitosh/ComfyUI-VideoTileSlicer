"""Pure geometry. Frame ranges are half-open; feather is not coverage overlap."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Settings:
    width: int = 1920
    height: int = 1088
    tile_width: int = 1024
    tile_height: int = 624
    frames: int = 33
    overlap_frames: int = 9
    fade_frames: int = 9
    overlap_x: int = 32
    overlap_y: int = 32
    feather_x: int = 32
    feather_y: int = 32
    layout: str = "workflow"
    alignment: str = "wan"
    end_policy: str = "pad"
    curve: str = "linear"
    fps: float = 24.0

    def validate(self):
        if min(self.width, self.height, self.tile_width, self.tile_height, self.frames) < 1:
            raise ValueError("Dimensions and segment length must be positive")
        if not 0 <= self.overlap_frames < self.frames:
            raise ValueError("Temporal overlap must be smaller than the segment length")
        if not 0 <= self.fade_frames <= self.overlap_frames:
            raise ValueError("Temporal fade must fit within the temporal overlap")
        for tile, overlap, feather in [(self.tile_width, self.overlap_x, self.feather_x),
                                      (self.tile_height, self.overlap_y, self.feather_y)]:
            if not 0 <= overlap < tile or not 0 <= feather < tile:
                raise ValueError("Overlap and feather must be nonnegative and smaller than a tile")
        if self.alignment not in ("wan", "none") or self.layout not in ("workflow", "minimum_overlap"):
            raise ValueError("Unknown alignment or tile layout")
        if self.end_policy not in ("pad", "trim") or self.curve not in ("linear", "smoothstep"):
            raise ValueError("Unknown end policy or curve")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("FPS must be finite and positive")
        if self.alignment == "wan":
            if (self.frames - 1) % 4 or self.tile_width % 16 or self.tile_height % 16:
                raise ValueError("Wan preset requires 4n+1 frames and tile dimensions divisible by 16")
        self.tiles()  # Validate actual overlaps too.
        return self

    def tiles(self):
        xs = axis(self.width, self.tile_width, self.overlap_x, self.layout)
        ys = axis(self.height, self.tile_height, self.overlap_y, self.layout)
        result = []
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                ox = xs[col - 1] + self.tile_width - x if col else 0
                oy = ys[row - 1] + self.tile_height - y if row else 0
                if (col and self.feather_x > ox) or (row and self.feather_y > oy):
                    raise ValueError("Feather exceeds actual tile overlap; reduce feather or use minimum_overlap layout")
                result.append(dict(x=x, y=y, width=self.tile_width, height=self.tile_height,
                                   valid_width=min(self.tile_width, self.width-x),
                                   valid_height=min(self.tile_height, self.height-y),
                                   row=row, col=col, overlap_x=ox, overlap_y=oy))
        return result

    def to_dict(self):
        return asdict(self)


def axis(size, tile, overlap, layout):
    if tile >= size:
        return [0]
    n = math.ceil(size / tile) if layout == "workflow" else math.ceil((size-overlap)/(tile-overlap))
    # Exact endpoints, integer coordinates, no uncovered edge due to rounding.
    return [round(i * (size-tile)/(n-1)) for i in range(n)]


def frame_range(start, total, settings):
    valid = min(settings.frames, total-start)
    if valid <= 0:
        raise ValueError("Segment starts past end of source")
    if settings.end_policy == "trim" and settings.alignment == "wan":
        valid = ((valid-1)//4)*4+1
    return valid, settings.frames if settings.end_policy == "pad" else valid

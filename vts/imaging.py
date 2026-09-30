"""CPU float32 preparation and compositing; no ComfyUI or model imports."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def cpu(images):
    return images.detach().to(device="cpu", dtype=torch.float32).contiguous()


def validate_images(images, name="images"):
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] != 3:
        raise ValueError(f"{name} must be a nonempty RGB IMAGE batch [T,H,W,3]")
    if not torch.isfinite(images).all():
        raise ValueError(f"{name} contains non-finite values")


def fingerprint(images):
    digest = hashlib.sha256(str(tuple(images.shape)).encode())
    for frame in images:
        digest.update(cpu(frame).numpy().tobytes())
    return digest.hexdigest()


def resize(images, width, height, mask=False):
    images = cpu(images)
    if images.shape[1:3] == (height, width):
        return images.clone()
    if mask:
        return F.interpolate(images[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0].clamp(0, 1)
    # Match the workflow's Lanczos resize. Keep same-size data at full precision.
    frames = []
    for frame in images:
        a = (frame.clamp(0, 1).numpy()*255).round().astype(np.uint8)
        a = np.asarray(Image.fromarray(a).resize((width, height), Image.Resampling.LANCZOS)).copy()
        frames.append(torch.from_numpy(a).float()/255)
    return torch.stack(frames)


def select(images, start, length, total, broadcast=False):
    if broadcast and len(images) == 1:
        return cpu(images).expand(length, *images.shape[1:]).clone()
    if len(images) != total:
        raise ValueError(f"Aligned input has {len(images)} frames; expected {total} (or a single broadcast frame)")
    indices = torch.arange(start, start+length, device=images.device).clamp(max=total-1)
    return cpu(images[indices])


def crop(images, tile):
    x, y, w, h = (tile[k] for k in ("x", "y", "width", "height"))
    out = images[:, y:y+h, x:x+w]
    dh, dw = h-out.shape[1], w-out.shape[2]
    if dh or dw:
        if out.ndim == 4:
            out = F.pad(out.permute(0, 3, 1, 2), (0, dw, 0, dh), mode="replicate").permute(0, 2, 3, 1)
        else:
            out = F.pad(out[:, None], (0, dw, 0, dh), mode="replicate")[:, 0]
    return out.clone()


def ramp(n, curve="linear"):
    # A one-frame fade chooses the new frame (no divide by zero).
    a = torch.linspace(0, 1, n) if n > 1 else torch.ones(n)
    return a*a*(3-2*a) if curve == "smoothstep" else a


def masks(tile, settings, length, incoming):
    h, w = tile["height"], tile["width"]
    protect = torch.ones(h, w)
    spatial = torch.ones(h, w)
    # Workflow protects/feathers only the configured border, not all coverage overlap.
    for dim, enabled, feather in [(1, tile["col"] > 0, settings.feather_x),
                                  (0, tile["row"] > 0, settings.feather_y)]:
        if enabled and feather:
            if dim == 1:
                protect[:, :feather] = 0
                # Coords gradient reaches 1 at x=feather, matching the source node.
                a = torch.arange(feather)/feather
                if settings.curve == "smoothstep":
                    a = a*a*(3-2*a)
                spatial[:, :feather] *= a[None, :]
            else:
                protect[:feather] = 0
                a = torch.arange(feather)/feather
                if settings.curve == "smoothstep":
                    a = a*a*(3-2*a)
                spatial[:feather] *= a[:, None]
    control = protect[None].expand(length, -1, -1).clone()
    control[:incoming] = 0
    blend = spatial[None].expand(length, -1, -1).clone()
    if incoming:
        fade = min(settings.fade_frames, incoming)
        # Hold preceding context first, then fade at the end of the shared range.
        temporal = torch.cat([torch.zeros(incoming-fade), ramp(fade, settings.curve)])
        blend[:incoming] *= temporal[:, None, None]
    return control, blend


def sharpen(images, amount):
    """Contrast-adaptive sharpening, matching essentials' CAS neighborhood."""
    if amount == 0:
        return images
    x = F.pad(images.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant")
    a,b,c = x[:,:,:-2,:-2],x[:,:,:-2,1:-1],x[:,:,:-2,2:]
    d,e,f = x[:,:,1:-1,:-2],x[:,:,1:-1,1:-1],x[:,:,1:-1,2:]
    g,h,i = x[:,:,2:,:-2],x[:,:,2:,1:-1],x[:,:,2:,2:]
    mn = torch.minimum(torch.minimum(torch.minimum(d,e),f),torch.minimum(b,h))
    mx = torch.maximum(torch.maximum(torch.maximum(d,e),f),torch.maximum(b,h))
    mn = mn + torch.minimum(torch.minimum(a,c),torch.minimum(g,i))
    mx = mx + torch.maximum(torch.maximum(a,c),torch.maximum(g,i))
    amp = (torch.minimum(mn,2-mx)/(mx+1e-5)).clamp_min(0).sqrt()
    weight = -amp*(amount*(1/5-1/8)+1/8)
    return ((weight*(b+d+f+h)+e)/(1+4*weight)).clamp(0,1).permute(0,2,3,1)


def reference(images, mode, transform, segment, tile_index, tile, settings, source):
    if images is None:
        images = source[:1]
        transform = "crop"
    else:
        validate_images(images, "reference")
        index = {"single": 0, "tile": tile_index, "segment": segment["index"], "frame": segment["start"]}[mode]
        if index >= len(images):
            raise ValueError(f"Reference mode {mode} needs index {index}, but only {len(images)} images supplied")
        images = cpu(images[index:index+1])
    if transform == "crop":
        return crop(resize(images, settings.width, settings.height), tile)
    if transform == "unchanged":
        return cpu(images)
    if transform == "fit":
        scale = max(tile["width"]/images.shape[2], tile["height"]/images.shape[1])
        out = resize(images, max(tile["width"], round(images.shape[2]*scale)), max(tile["height"], round(images.shape[1]*scale)))
        x, y = (out.shape[2]-tile["width"])//2, (out.shape[1]-tile["height"])//2
        return out[:, y:y+tile["height"], x:x+tile["width"]]
    return resize(images, tile["width"], tile["height"])

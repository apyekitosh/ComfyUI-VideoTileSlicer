"""VideoTileSlicer 2: model-independent, resumable video tile preparation."""
from comfy_api.latest import ComfyExtension
from .nodes import NODES

WEB_DIRECTORY = "./js"


class VideoTileSlicerExtension(ComfyExtension):
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint():
    return VideoTileSlicerExtension()

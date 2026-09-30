import ast
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from vts.imaging import sharpen, masks
from vts.geometry import Settings, axis


def test_cas_matches_installed_workflow_node():
    source=Path('C:/ComfyUI/custom_nodes/ComfyUI_essentials/image.py')
    if not source.exists():
        pytest.skip('Optional local source parity check')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ImageCAS')
    namespace=dict(torch=torch,F=F,min_=lambda x: torch.stack(x).amin(0),max_=lambda x: torch.stack(x).amax(0))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'ImageCAS','exec'),namespace)
    images=torch.rand(5,16,16,3)
    expected=namespace['ImageCAS']().execute(images,.8)[0]
    assert torch.allclose(sharpen(images,.8),expected,atol=2e-6)


@pytest.mark.parametrize('size,tile,overlap',[(1920,1024,32),(100,32,8),(33,32,0),(64,32,0),(7,16,2)])
def test_axis_covers_edges(size,tile,overlap):
    starts=axis(size,tile,overlap,'minimum_overlap')
    coverage=torch.zeros(size)
    for x in starts:
        coverage[x:min(size,x+tile)]+=1
    assert coverage.min()>=1
    assert starts[0]==0 and starts[-1]+tile>=size
    for a,b in zip(starts,starts[1:]):
        assert a+tile-b>=overlap


def test_one_frame_fade_and_distinct_width():
    s=Settings(width=16,height=16,tile_width=16,tile_height=16,frames=5,
               overlap_frames=3,fade_frames=1,overlap_x=0,overlap_y=0,feather_x=0,feather_y=0)
    control,blend=masks(s.tiles()[0],s,5,3)
    assert torch.equal(blend[:,0,0],torch.tensor([0.,0.,1.,1.,1.]))
    assert torch.equal(control[:,0,0],torch.tensor([0.,0.,0.,1.,1.]))


def test_invalid_alignment_and_feather():
    with pytest.raises(ValueError,match='Wan'):
        Settings(frames=54).validate()
    with pytest.raises(ValueError,match='actual'):
        Settings(width=2048,feather_x=32).validate()

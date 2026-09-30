from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import uuid
import pytest
import torch
from vts.engine import Job, prepare, commit
from vts.geometry import Settings
from vts import storage as disk, imaging


def make_job(tmp_path):
    images=torch.rand(11,16,16,3)
    s=Settings(width=16,height=16,tile_width=16,tile_height=16,frames=5,overlap_frames=1,fade_frames=1,
               overlap_x=0,overlap_y=0,feather_x=0,feather_y=0)
    return Job(str(tmp_path/'job'),images,imaging.fingerprint(images),json.dumps(s.to_dict()))


def test_orphan_recovery_preserves_checkpoint_and_history(tmp_path):
    j=make_job(tmp_path)
    p=prepare(j,sharpen=0)
    state=disk.read(j.path)
    orphan=disk.save_checkpoint(j.path,torch.zeros(1,2,2,3))
    orphan_seq='sequence/'+uuid.uuid4().hex
    disk.path(j.path,orphan_seq).mkdir(parents=True)
    again=prepare(j,sharpen=0)
    assert again['ticket']==p['ticket']
    assert not disk.path(j.path,orphan).exists()
    assert not disk.path(j.path,orphan_seq).exists()
    assert disk.path(j.path,state['revisions'][state['active']]['checkpoint']).exists()


def test_larger_overlap_uses_raw_not_only_trailing_folder(tmp_path):
    j=make_job(tmp_path)
    p=prepare(j,sharpen=0)
    commit(p['ticket'],p['source'])
    j=replace(j,settings_json=json.dumps(replace(j.settings,overlap_frames=3,fade_frames=2).to_dict()))
    p=prepare(j,sharpen=0)
    assert p['info']['active']['start']==2
    assert p['info']['active']['incoming']==3
    assert p['mask'][:3].sum()==0


def test_raw_write_failure_does_not_publish(tmp_path,monkeypatch):
    j=make_job(tmp_path)
    p=prepare(j,sharpen=0)
    before=disk.read(j.path)
    original=disk.save_png
    counter=0
    def fail(path,frame):
        nonlocal counter
        counter+=1
        if counter==3:
            raise OSError('mid-frame-write failure')
        original(path,frame)
    monkeypatch.setattr(disk,'save_png',fail)
    with pytest.raises(OSError):
        commit(p['ticket'],p['source'])
    assert disk.read(j.path)==before
    assert not list((Path(j.path)/'segments').glob('*/revisions/*/*'))


def test_os_lock_rejects_second_process(tmp_path):
    root=Path(__file__).resolve().parents[1]
    code='''import sys
sys.path.insert(0,sys.argv[1])
from vts.storage import locked
try:
    with locked(sys.argv[2]):
        raise SystemExit(1)
except RuntimeError:
    raise SystemExit(0)
'''
    with disk.locked(tmp_path):
        result=subprocess.run([sys.executable,'-c',code,str(root),str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_old_folder_is_not_modified(tmp_path):
    j=make_job(tmp_path)
    Path(j.path).mkdir()
    old=Path(j.path)/'jobinfo.json'
    old.write_text('{"legacy":true}')
    with pytest.raises(ValueError,match='empty folder'):
        prepare(j,sharpen=0)
    assert old.read_text()=='{"legacy":true}'
    assert not (Path(j.path)/'job.json').exists()

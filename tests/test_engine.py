from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from vts.geometry import Settings
from vts.engine import Job, prepare, commit, preview, restore_revision, describe
from vts import imaging as im, storage as disk


def settings(**kwargs):
    defaults = dict(width=12, height=10, tile_width=8, tile_height=6, frames=5,
                    overlap_frames=2, fade_frames=2, overlap_x=2, overlap_y=2,
                    feather_x=2, feather_y=2, alignment="none")
    return Settings(**{**defaults, **kwargs}).validate()


def job(tmp_path, frames=12, **kwargs):
    source = torch.linspace(0, 1, frames*10*12*3).reshape(frames,10,12,3)
    return Job(str(tmp_path/'job'), source, im.fingerprint(source), json.dumps(settings(**kwargs).to_dict()))


def prep(j, **kwargs):
    return prepare(j, sharpen=0, **kwargs)


def finish_segment(j, value=None):
    for _ in range(30):
        p = prep(j)
        assert p is not None
        result = commit(p['ticket'], p['source'] if value is None else torch.full_like(p['source'], value))
        if result['segment_complete']:
            return result
    raise AssertionError('did not complete')


def test_workflow_geometry_and_separate_feather():
    s = Settings().validate()
    tiles = s.tiles()
    assert [(t['x'],t['y']) for t in tiles] == [(0,0),(896,0),(0,464),(896,464)]
    assert tiles[-1]['overlap_x'] == 128
    assert tiles[-1]['overlap_y'] == 160
    mask, blend = im.masks(tiles[-1],s,33,9)
    assert mask[9,32,32] == 1 and mask[9,31,32] == 0
    assert blend[9,32,32] == 1
    assert blend[9,16,16] == .25
    assert blend[0].sum() == 0
    assert torch.allclose(blend[8],blend[9])


def test_progress_only_on_commit_and_resume(tmp_path):
    j = job(tmp_path)
    p = prep(j)
    state = disk.read(j.path)
    again = prep(j)
    assert state == disk.read(j.path)
    assert p['ticket'] == again['ticket']
    assert torch.equal(p['control'],again['control'])
    result = commit(p['ticket'],p['source'])
    assert not result['segment_complete']
    assert prep(j)['ticket'].tile == 1
    assert len(list((Path(j.path)/'checkpoints').glob('*.npy'))) == 1


def test_duplicate_stale_and_wrong_shape(tmp_path):
    j = job(tmp_path)
    p = prep(j)
    with pytest.raises(ValueError,match='shape'):
        commit(p['ticket'],p['source'][:1])
    state = disk.read(j.path)
    commit(p['ticket'],p['source'])
    after = disk.read(j.path)
    assert commit(p['ticket'],p['source'])['duplicate']
    assert after == disk.read(j.path)
    q = prep(j)
    with pytest.raises(ValueError,match='Stale'):
        commit(replace(q['ticket'],version=state['version']),q['source'])


def test_commit_interruption_rolls_back(tmp_path,monkeypatch):
    j = job(tmp_path)
    p = prep(j)
    before = disk.read(j.path)
    original = disk.publish
    def fail(*args):
        raise OSError('injected disk failure')
    monkeypatch.setattr(disk,'publish',fail)
    with pytest.raises(OSError):
        commit(p['ticket'],torch.ones_like(p['source']))
    assert disk.read(j.path) == before
    assert len(list((Path(j.path)/'checkpoints').glob('*.npy'))) == 1
    monkeypatch.setattr(disk,'publish',original)
    assert torch.equal(prep(j)['control'],p['control'])
    commit(p['ticket'],p['source'])


def test_variable_lengths_freeze_and_complete(tmp_path):
    j = job(tmp_path,frames=240, width=12,height=10,tile_width=12,tile_height=10,
            overlap_frames=2,fade_frames=2)
    # Use single tiles but Wan temporal sizes to verify requested 81/53/121.
    ranges = []
    for length in [81,53,121]:
        j = replace(j,settings_json=json.dumps(replace(j.settings,frames=length,overlap_frames=9,fade_frames=9).to_dict()))
        p = prep(j)
        ranges.append((p['info']['active']['start'],p['info']['active']['length']))
        changed = replace(j,settings_json=json.dumps(replace(j.settings,frames=17).to_dict()))
        assert len(prep(changed)['source']) == length
        commit(p['ticket'],p['source'])
    assert ranges == [(0,81),(72,53),(116,121)]
    finish_segment(j)
    assert prep(j) is None
    files = list((Path(j.path)/disk.read(j.path)['sequence']).glob('*.png'))
    assert len(files)==240


def test_revisions_keep_raw_and_restore(tmp_path):
    j = job(tmp_path,frames=8)
    finish_segment(j,.2)
    finish_segment(j,.8)
    state = disk.read(j.path)
    original = state['selected']['0']
    raw0 = disk.raw_frames(j.path,state['revisions'][original],0,5).clone()
    original_sequence,_ = preview(j.path,count=20)
    edit = replace(j, action='revise',segment=0,request_id=1)
    p = prep(edit)
    assert disk.read(j.path)['selected']['0'] == original
    commit(p['ticket'],torch.full_like(p['source'],.4))
    finish_segment(edit,.4)
    assert prep(edit) is None
    state = disk.read(j.path)
    assert state['selected']['0'] != original
    assert describe(state)['selected'][1]['stale_context']
    assert torch.equal(disk.raw_frames(j.path,state['revisions'][original],0,5),raw0)
    restore_revision(j.path,0,1,2)
    restored,_ = preview(j.path,count=20)
    assert torch.equal(restored,original_sequence)
    assert not describe(disk.read(j.path))['selected'][1]['stale_context']
    assert len(list((Path(j.path)/'sequence').iterdir())) == 1


def test_cascade_rebuilds_context(tmp_path):
    j = job(tmp_path,frames=8)
    finish_segment(j,.2)
    finish_segment(j,.8)
    edit = replace(j,action='regenerate_from',segment=0,request_id=12)
    finish_segment(edit,.4)
    finish_segment(edit,.7)
    assert prep(edit) is None
    state = disk.read(j.path)
    assert len(state['revisions']) == 4
    assert not any(r['stale_context'] for r in describe(state)['selected'])


def test_reference_selection_and_validation_does_not_freeze_job(tmp_path):
    j = job(tmp_path)
    with pytest.raises(ValueError):
        prep(j,control_mode='supplied')
    assert disk.read(j.path) is None
    refs = torch.stack([torch.full((10,12,3),i/4) for i in range(4)])
    p = prep(j,references=refs)
    assert p['reference'].mean()==0
    commit(p['ticket'],p['source'])
    p = prep(j,references=refs)
    assert p['reference'].mean()==.25
    with pytest.raises(ValueError,match='changed'):
        prep(j,references=refs+.01)


def test_preview_is_read_only_and_bounded(tmp_path):
    j = job(tmp_path)
    prep(j)
    before = disk.read(j.path)
    images,_ = preview(j.path,'active',start=1,count=2)
    assert len(images)==2 and disk.read(j.path)==before


def test_final_commit_publication_failure_keeps_previous_sequence(tmp_path,monkeypatch):
    j = job(tmp_path,frames=8)
    finish_segment(j)
    for i in range(3):
        p=prep(j)
        commit(p['ticket'],p['source'])
    p=prep(j)
    before = disk.read(j.path)
    images,_ = preview(j.path,count=20)
    def fail(*args):
        raise OSError('publish failed')
    monkeypatch.setattr(disk,'publish',fail)
    with pytest.raises(OSError):
        commit(p['ticket'],p['source'])
    assert disk.read(j.path)==before
    assert torch.equal(preview(j.path,count=20)[0],images)


def test_single_tile_padding_and_mask_broadcast(tmp_path):
    j=job(tmp_path,frames=3,tile_width=16,tile_height=16)
    p=prep(j,user_mask=torch.ones(1,10,12))
    assert p['source'].shape==(5,16,16,3)
    assert torch.equal(p['source'][2],p['source'][4])
    commit(p['ticket'],p['source'])
    assert prep(j,user_mask=torch.ones(1,10,12)) is None
    out,_=preview(j.path,count=10)
    assert out.shape==(3,10,12,3)


def test_source_change_rejected(tmp_path):
    j=job(tmp_path)
    prep(j)
    with pytest.raises(ValueError,match='Source changed'):
        prep(replace(j,source_hash='other'))


def test_restart_after_first_tile_uses_new_settings_once(tmp_path):
    j = job(tmp_path)
    first = prep(j)
    commit(first['ticket'], torch.ones_like(first['source']))
    pending = prep(j)
    old = disk.read(j.path)['revisions'][pending['ticket'].revision]
    changed = replace(j.settings, frames=7, tile_width=10, feather_x=1)
    restart = replace(j, action='restart_active', request_id=10, segment=99,
                      settings_json=json.dumps(changed.to_dict()))
    p = prep(restart, control_mode='neutral')
    assert p['ticket'].tile == 0
    assert p['ticket'].revision != pending['ticket'].revision
    assert p['source'].shape == (7, 6, 10, 3)
    assert torch.equal(p['context'], p['source'])  # No pixels from abandoned tiles.
    state = disk.read(j.path)
    assert state['revisions'][old['key']]['status'] == 'abandoned'
    assert not disk.path(j.path, old['checkpoint']).exists()
    assert len(list((Path(j.path)/'checkpoints').glob('*.npy'))) == 1
    assert prep(restart, control_mode='neutral')['ticket'] == p['ticket']
    with pytest.raises(ValueError, match='Stale'):
        commit(pending['ticket'], pending['source'])
    commit(p['ticket'], p['source'])
    assert prep(restart, control_mode='neutral')['ticket'].tile == 1
    for _ in range(3):
        p = prep(restart, control_mode='neutral')
        result = commit(p['ticket'], p['source'])
    assert result['segment_complete'] and not result['continue_run']
    assert prep(restart, control_mode='neutral') is None
    assert prep(replace(restart, action='continue'), control_mode='neutral')['info']['active']['index'] == 1


def test_restart_before_first_commit_and_second_explicit_restart(tmp_path):
    j = job(tmp_path)
    original = prep(j)
    restart = replace(j, action='restart_active', request_id=1)
    p = prep(restart)
    assert p['ticket'].tile == 0
    with pytest.raises(ValueError, match='Stale'):
        commit(original['ticket'], original['source'])
    again = prep(replace(restart, request_id=2))
    assert again['ticket'].revision != p['ticket'].revision
    assert again['ticket'].tile == 0
    assert len(list((Path(j.path)/'checkpoints').glob('*.npy'))) == 1


@pytest.mark.parametrize('failure', ['invalid_control', 'publish'])
def test_failed_restart_preserves_original_attempt(tmp_path, monkeypatch, failure):
    j = job(tmp_path)
    p = prep(j)
    commit(p['ticket'], p['source'])
    pending = prep(j)
    before = disk.read(j.path)
    restart = replace(j, action='restart_active', request_id=1)
    if failure == 'publish':
        original = disk.publish
        def fail(*args):
            raise OSError('restart publication failed')
        monkeypatch.setattr(disk, 'publish', fail)
        with pytest.raises(OSError):
            prep(restart)
        monkeypatch.setattr(disk, 'publish', original)
    else:
        with pytest.raises(ValueError, match='Aligned input'):
            prep(restart, control_mode='supplied', control=j.source[:2])
    assert disk.read(j.path) == before
    assert prep(j)['ticket'] == pending['ticket']
    assert len(list((Path(j.path)/'checkpoints').glob('*.npy'))) == 1


def test_restart_new_segment_recalculates_range_and_preserves_export(tmp_path):
    j = job(tmp_path)
    finish_segment(j)
    p = prep(j)
    commit(p['ticket'], p['source'])
    before = disk.read(j.path)
    exported, _ = preview(j.path, count=20)
    changed = replace(j.settings, frames=7, overlap_frames=3, fade_frames=1)
    restart = replace(j, action='restart_active', request_id=1,
                      settings_json=json.dumps(changed.to_dict()))
    p = prep(restart)
    assert (p['info']['active']['start'], len(p['source'])) == (2, 7)
    after = disk.read(j.path)
    assert after['sequence'] == before['sequence']
    assert after['selected'] == before['selected']
    assert torch.equal(preview(j.path, count=20)[0], exported)
    assert p['mask'][:3].sum() == 0


def test_restart_historical_revision_keeps_range_and_cascade(tmp_path):
    j = job(tmp_path, frames=8)
    finish_segment(j)
    finish_segment(j)
    edit = replace(j, action='regenerate_from', segment=0, request_id=1)
    p = prep(edit)
    commit(p['ticket'], p['source'])
    changed = replace(j.settings, frames=7, tile_width=10)
    restart = replace(j, action='restart_active', request_id=2,
                      settings_json=json.dumps(changed.to_dict()))
    p = prep(restart)
    assert (p['info']['active']['start'], len(p['source'])) == (0, 5)
    assert p['source'].shape[2] == 10
    assert disk.read(j.path)['cascade'] == [1]
    finish_segment(restart)
    finish_segment(restart)
    assert prep(restart) is None
    assert not any(s['stale_context'] for s in describe(disk.read(j.path))['selected'])


def test_restart_requires_active_segment_and_unused_request_id(tmp_path):
    j = job(tmp_path, frames=5)
    restart = replace(j, action='restart_active', request_id=1)
    with pytest.raises(ValueError, match='No active'):
        prep(restart)
    finish_segment(j)
    with pytest.raises(ValueError, match='No active'):
        prep(restart)
    prep(replace(j, action='revise', request_id=1))
    before = disk.read(j.path)
    with pytest.raises(ValueError, match='already used'):
        prep(restart)
    assert disk.read(j.path) == before

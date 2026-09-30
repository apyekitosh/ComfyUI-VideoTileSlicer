"""Optional real-ComfyUI CPU integration check; no server listener or models."""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile

parser=argparse.ArgumentParser()
parser.add_argument('--comfy-root',required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(args.comfy_root).resolve()))
sys.argv=['vts-api-check','--cpu']
import nodes
import execution
import folder_paths

ROOT=Path(__file__).resolve().parents[1]


class Events:
    client_id='vts-test'
    last_node_id=None
    def __init__(self):
        self.messages=[]
    def send_sync(self,event,data,sid=None):
        self.messages.append((event,data))


async def main():
    assert await nodes.load_custom_node(str(ROOT))
    for name in ('VTS2_Job','VTS2_Prepare','VTS2_Commit','VTS2_Preview'):
        assert name in nodes.NODE_CLASS_MAPPINGS
        nodes.NODE_CLASS_MAPPINGS[name].GET_SCHEMA()
    with tempfile.TemporaryDirectory(prefix='vts-comfy-integration-') as tmp:
        folder_paths.set_temp_directory(tmp)
        job=Path(tmp)/'job'
        def defaults(name):
            return {i.id:i.default for i in nodes.NODE_CLASS_MAPPINGS[name].GET_SCHEMA().inputs if hasattr(i,'default')}
        job_inputs={**defaults('VTS2_Job'), 'source':['1',0],'job_folder':str(job),
                    'width':32,'height':32,'tile_width':16,'tile_height':16,'frames':5,
                    'overlap_frames':1,'fade_frames':1,'overlap_x':0,'overlap_y':0,'feather_x':0,'feather_y':0}
        prompt={
            '1':{'class_type':'EmptyImage','inputs':dict(width=32,height=32,batch_size=9,color=0x557799)},
            '2':{'class_type':'VTS2_Job','inputs':job_inputs},
            '3':{'class_type':'VTS2_Prepare','inputs':{**defaults('VTS2_Prepare'),'job':['2',0],'sharpen':0}},
            '4':{'class_type':'VTS2_Commit','inputs':{**defaults('VTS2_Commit'),'generated':['3',0],'ticket':['3',5]}},
        }
        events=Events()
        executor=execution.PromptExecutor(events,cache_type=execution.CacheType.CLASSIC,cache_args={"ram": 0})
        # Start a segment, then reject its first tile and restart with changed
        # preparation settings through the real cached graph execution path.
        p=copy.deepcopy(prompt)
        valid=await execution.validate_prompt('initial-attempt',p,None)
        assert valid[0],valid
        await executor.execute_async(p,'initial-attempt',{'client_id':'vts-test'},valid[2])
        assert executor.success,executor.status_messages
        initial=json.loads((job/'job.json').read_text())
        abandoned=initial['active']
        prompt['2']['inputs'].update(action='restart_active',request_id=1)
        prompt['3']['inputs']['sharpen']=0.2
        for i in range(8):
            if i == 4:
                prompt['2']['inputs']['action']='continue'
            p=copy.deepcopy(prompt)
            valid=await execution.validate_prompt(str(i),p,None)
            assert valid[0],valid
            await executor.execute_async(p,str(i),{'client_id':'vts-test'},valid[2])
            assert executor.success,executor.status_messages
            state=json.loads((job/'job.json').read_text())
            assert state['revisions'][abandoned]['status']=='abandoned'
            if state['active']:
                assert state['revisions'][state['active']]['prep']['sharpen']==0.2
            print("run", i, "active", state["active"], "version", state["version"], "tiles", [r["next_tile"] for r in state["revisions"].values()])
            if i%4 != 3:
                assert state['revisions'][state['active']]['next_tile']==(i%4)+1
            else:
                assert len(state['selected'])==(i//4)+1
        assert len(list((job/state['sequence']).glob('*.png')))==9
        # Queueing a completed job cleanly blocks dependent sampling/commit.
        await executor.execute_async(copy.deepcopy(prompt),'done',{'client_id':'vts-test'},['4'])
        assert executor.success
        p={'5':{'class_type':'VTS2_Preview','inputs':{**defaults('VTS2_Preview'),'job_folder':str(job),'count':9}}}
        valid=await execution.validate_prompt('preview',p,None)
        assert valid[0],valid
        await executor.execute_async(p,'preview',{'client_id':'vts-test'},valid[2])
        assert executor.success,executor.status_messages
        assert any(event=='executed' and data['output'].get('images') for event,data in events.messages)
        print('Real ComfyUI: registration, active restart with changed settings, eight cached tile executions, completion blocking and previews passed.')


asyncio.run(main())

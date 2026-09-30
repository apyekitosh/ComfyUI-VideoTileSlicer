import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const handlers = new Map();
let extension, queued=0, queue={Pending:[],Running:[]};
const node={id:3,comfyClass:'VTS2_Commit',inputs:[],widgets:[{name:'auto_queue',value:'continuous'}]};
const graph={_nodes:[node],getNodeById:()=>node};
const app={graph,registerExtension:x=>extension=x,queuePrompt:async()=>{queued++;},extensionManager:{toast:{add(){}}}};
const api={addEventListener:(n,f)=>handlers.set(n,f),getQueue:async()=>queue};
const source=readFileSync(new URL('../js/autoqueue.js',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'');
vm.runInNewContext(source,{app,api,console,Map,Set,JSON});
extension.setup();
async function emit(name,detail){await handlers.get(name)?.({detail});}
async function start(id,nonce){
 await emit('execution_start',{prompt_id:id});
 await emit('executed',{prompt_id:id,node:3,output:{vts2:[{nonce,continue_run:true}]}});
}
await start('one','one');
assert.equal(queued,0,'must wait for entire prompt');
await emit('execution_success',{prompt_id:'one'});
assert.equal(queued,1);
await emit('execution_success',{prompt_id:'one'});
assert.equal(queued,1,'duplicate success');
await start('two','two');
await emit('execution_error',{prompt_id:'two'});
await emit('execution_success',{prompt_id:'two'});
assert.equal(queued,1,'downstream failure');
await start('three','three');
node.widgets[0].value='off';
await emit('execution_success',{prompt_id:'three'});
assert.equal(queued,1,'pause');
node.widgets[0].value='continuous';
await start('four','four');
app.graph={...graph};
await emit('execution_success',{prompt_id:'four'});
assert.equal(queued,1,'switched graph');
app.graph=graph;
await start('five','five');
queue.Pending=[['some work']];
await emit('execution_success',{prompt_id:'five'});
assert.equal(queued,1,'existing pending work');
console.log('Autoqueue: success, deduplication, failure, pause, graph-switch and queue guards passed.');

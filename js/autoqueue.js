import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const pending = new Map();
const starts = new Map();
const seen = new Set();
function signature() {
    return JSON.stringify((app.graph?._nodes ?? []).map(n => ({
        id: n.id, type: n.comfyClass, inputs: n.inputs?.map(i => i.link),
        widgets: n.widgets?.filter(w => w.options?.serialize !== false && w.type !== "button").map(w => [w.name, w.value])
    })));
}
app.registerExtension({
    name: "VTS2.Autoqueue",
    setup() {
        api.addEventListener("execution_start", ({ detail }) => {
            starts.set(detail.prompt_id, { graph: app.graph, signature: signature() });
        });
        api.addEventListener("executed", ({ detail }) => {
            const node = app.graph?.getNodeById(detail.node);
            const result = detail.output?.vts2?.[0];
            if (node?.comfyClass !== "VTS2_Commit" || !result?.continue_run || seen.has(result.nonce)) return;
            pending.set(detail.prompt_id, { node, nonce: result.nonce });
        });
        for (const event of ["execution_error", "execution_interrupted"]) {
            api.addEventListener(event, ({ detail }) => {
                pending.delete(detail.prompt_id);
                starts.delete(detail.prompt_id);
            });
        }
        api.addEventListener("execution_success", async ({ detail }) => {
            const next = pending.get(detail.prompt_id);
            const started = starts.get(detail.prompt_id);
            pending.delete(detail.prompt_id);
            starts.delete(detail.prompt_id);
            if (!next || !started || started.graph !== app.graph || started.signature !== signature()) return;
            if (!app.graph._nodes.includes(next.node) || next.node.widgets.find(w => w.name === "auto_queue")?.value === "off") return;
            if (seen.has(next.nonce)) return;
            seen.add(next.nonce);
            if (seen.size > 1000) seen.delete(seen.values().next().value);
            try {
                const queue = await api.getQueue();
                if (queue.Pending?.length || queue.Running?.some(p => p[1] !== detail.prompt_id)) return;
                if (started.graph !== app.graph || started.signature !== signature()) return;
                await app.queuePrompt(0, 1);
            } catch (error) {
                app.extensionManager.toast.add({ severity: "error", summary: "VTS autoqueue paused", detail: String(error) });
            }
        });
    },
    nodeCreated(node) {
        if (node.comfyClass !== "VTS2_Commit") return;
        node.addWidget("button", "Pause after current tile", null, () => {
            const widget = node.widgets.find(w => w.name === "auto_queue");
            if (widget) widget.value = "off";
            pending.clear();
        }, { serialize: false });
    }
});

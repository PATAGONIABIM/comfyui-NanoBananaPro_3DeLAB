import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.NanoBananaPro",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiNanoBananaPro") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.bgcolor = "#940000";
                return r;
            };
        }
    },
});

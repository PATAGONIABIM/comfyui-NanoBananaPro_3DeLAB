import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.NanoBananaPro",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiNanoBananaPro" ||
            nodeData.name === "LoadScribbleImage" ||
            nodeData.name === "GeminiVeo31VideoGenerator" ||
            nodeData.name === "NanoBananaPreviewVideo" ||
            nodeData.name === "LoadVideoExtract_3DeLAB" ||
            nodeData.name === "ImagePassthrough_3DeLAB") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.bgcolor = "#2d0000ff";
                this.color = "#000000";
                return r;
            };
        }
    },
});

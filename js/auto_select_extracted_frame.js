import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "NanoBananaPro.AutoSelectFrame",
    async setup() {
        api.addEventListener("nanobanana_video_extracted", (event) => {
            const filename = event.detail.filename;
            console.log(`[NanoBananaPro] Received video extraction event for: ${filename}`);

            // Search all active nodes in the ComfyUI Graph
            for (const node of app.graph._nodes) {
                if (node.comfyClass === "ImagePassthrough_3DeLAB") {
                    const imageWidget = node.widgets.find(w => w.name === "image");
                    if (imageWidget) {

                        // Push to options if not already there so it can be selected
                        if (!imageWidget.options.values.includes(filename)) {
                            imageWidget.options.values.push(filename);
                        }

                        // Update the dropdown value to the new file
                        imageWidget.value = filename;

                        // Force UI refresh if it has a callback
                        if (imageWidget.callback) {
                            imageWidget.callback(filename);
                        }
                        console.log(`[NanoBananaPro] Auto-selected extracted frame in node ${node.id}`);
                    }
                }
            }
        });
    }
});

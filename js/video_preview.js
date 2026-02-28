import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "NanoBananaPro.VideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "NanoBananaPreviewVideo") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (message.video && message.video.length > 0) {
                    const videoData = message.video[0];
                    const filename = videoData.filename;
                    const subfolder = videoData.subfolder || "";
                    const type = videoData.type || "output";

                    // Create URL correctly encoded for ComfyUI's /view endpoint
                    let src = `/view?filename=${encodeURIComponent(filename)}&type=${type}`;
                    if (subfolder) {
                        src += `&subfolder=${encodeURIComponent(subfolder)}`;
                    }
                    src += `&t=${Date.now()}`;

                    let videoWidget = this.widgets?.find(w => w.name === "video_preview_widget");

                    if (!videoWidget) {
                        // Create HTML element
                        const videoEl = document.createElement("video");
                        videoEl.autoplay = true;
                        videoEl.loop = true;
                        videoEl.controls = true;
                        videoEl.muted = true; // Essential for autoplay in many browsers
                        videoEl.style.width = "100%";
                        videoEl.style.height = "auto";
                        videoEl.style.minHeight = "200px";

                        // ComfyUI magic to place DOM elements on the canvas
                        videoWidget = this.addDOMWidget("video_preview_widget", "video", videoEl, {
                            serialize: false,
                            hideOnZoom: false
                        });
                    }

                    // Update src
                    videoWidget.element.src = src;

                    // Wait for video metadata to trigger node resize wrapper
                    videoWidget.element.onloadeddata = () => {
                        requestAnimationFrame(() => {
                            const sz = this.computeSize();
                            if (sz[0] < this.size[0]) sz[0] = this.size[0];
                            if (sz[1] < this.size[1]) sz[1] = this.size[1];
                            this.onResize?.(sz);
                            app.graph.setDirtyCanvas(true, true);
                        });
                    };
                }
            };
        }
    },
});

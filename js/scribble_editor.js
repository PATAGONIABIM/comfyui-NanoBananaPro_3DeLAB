import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "NanoBananaPro.ScribbleEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "LoadScribbleImage") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                let r;
                try {
                    r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                } catch (e) {
                    console.error("[ScribbleEditor] Error in original onNodeCreated:", e);
                }

                try {
                    this.bgcolor = "#2d0000ff";
                    this.color = "#000000";

                    let scribbleDataWidget = this.widgets ? this.widgets.find(w => w.name === "scribble_data") : null;
                    if (!scribbleDataWidget) {
                        scribbleDataWidget = this.addWidget("string", "scribble_data", "", () => { }, { hidden: true });
                    }

                    const existingBtn = this.widgets ? this.widgets.find(w => w.name === "✏️ Abrir Scribble Editor") : null;
                    if (!existingBtn) {
                        this.addWidget("button", "✏️ Abrir Scribble Editor", "edit", () => {
                            openScribbleEditor(this, scribbleDataWidget);
                        });
                    }
                } catch (e) {
                    console.error("[ScribbleEditor] Error adding UI elements:", e);
                }

                return r;
            };

            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function (ctx) {
                try {
                    if (onDrawBackground) {
                        onDrawBackground.apply(this, arguments);
                    }

                    let scribbleDataWidget = this.widgets ? this.widgets.find(w => w.name === "scribble_data") : null;
                    if (scribbleDataWidget && scribbleDataWidget.value && scribbleDataWidget.value.startsWith("data:image")) {
                        if (this.imageRects && this.imageRects.length > 0) {
                            const rect = this.imageRects[0];

                            // Cache the image to avoid reloading it every frame
                            if (!this._cachedScribbleStr || this._cachedScribbleStr !== scribbleDataWidget.value) {
                                this._cachedScribbleImg = new Image();
                                this._cachedScribbleImg.onload = () => {
                                    this.setDirtyCanvas(true, false);
                                };
                                this._cachedScribbleImg.src = scribbleDataWidget.value;
                                this._cachedScribbleStr = scribbleDataWidget.value;
                            }

                            if (this._cachedScribbleImg && this._cachedScribbleImg.complete && this._cachedScribbleImg.naturalWidth > 0) {
                                ctx.drawImage(this._cachedScribbleImg, ...rect);
                            }
                        }
                    }
                } catch (e) {
                    console.error("[ScribbleEditor] Error in onDrawBackground:", e);
                }
            };
        }
    }
});

function openScribbleEditor(node, dataWidget) {
    const imageWidget = node.widgets.find(w => w.name === "image");
    const imageName = imageWidget ? imageWidget.value : null;

    if (!imageName) {
        alert("Por favor, selecciona una imagen base primero.");
        return;
    }

    // Modal UI Container
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", top: "0", left: "0", width: "100%", height: "100%",
        backgroundColor: "rgba(0,0,0,0.8)", zIndex: "9999",
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        overflow: "hidden"
    });

    const container = document.createElement("div");
    Object.assign(container.style, {
        backgroundColor: "#222", padding: "20px", borderRadius: "8px",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 4px 12px rgba(0,0,0,0.5)", position: "relative"
    });

    // --- Toolbar ---
    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, { display: "flex", gap: "10px", alignItems: "center", color: "white" });

    let activeTool = "pencil"; // pencil | eraser | text | arrow | move
    let brushSize = 10;
    let brushShape = "round"; // round | square
    const primaryColor = "#FF0000"; // Fixed to red

    const getIconUrl = (filename) => new URL(`assets/${filename}`, import.meta.url).href;

    // Tool Buttons
    const btnMove = document.createElement("button"); btnMove.innerHTML = `<img src="${getIconUrl('cursor.png')}" style="width:24px;height:24px;filter:invert(1);">`; btnMove.title = "Seleccionar y Mover";
    const btnPencil = document.createElement("button"); btnPencil.innerHTML = `<img src="${getIconUrl('lapiz.png')}" style="width:24px;height:24px;filter:invert(1);">`; btnPencil.title = "Lápiz (Rojo)"; btnPencil.style.background = "#555";
    const btnEraser = document.createElement("button"); btnEraser.innerHTML = `<img src="${getIconUrl('borrador.png')}" style="width:24px;height:24px;filter:invert(1);">`; btnEraser.title = "Borrador";
    const btnText = document.createElement("button"); btnText.innerHTML = `<img src="${getIconUrl('texto.png')}" style="width:24px;height:24px;filter:invert(1);">`; btnText.title = "Texto Libre (Rojo)";
    const btnArrow = document.createElement("button"); btnArrow.innerHTML = `<img src="${getIconUrl('flechas_boton.png')}" style="width:24px;height:24px;filter:invert(1);">`; btnArrow.title = "Flecha (Rojo)";

    // Common Button styling
    const tools = [btnMove, btnPencil, btnEraser, btnText, btnArrow];
    tools.forEach(btn => {
        Object.assign(btn.style, {
            width: "40px", height: "40px", cursor: "pointer",
            display: "flex", justifyContent: "center", alignItems: "center",
            background: "transparent", border: "1px solid #444", borderRadius: "4px"
        });
    });

    const activeColor = "#666";
    function setActiveTool(tool) {
        activeTool = tool;
        tools.forEach(b => b.style.background = "transparent");
        if (tool === "move") btnMove.style.background = activeColor;
        else if (tool === "pencil") btnPencil.style.background = activeColor;
        else if (tool === "eraser") btnEraser.style.background = activeColor;
        else if (tool === "text") btnText.style.background = activeColor;
        else if (tool === "arrow") btnArrow.style.background = activeColor;

        // Update canvas cursor setting
        if (tool === "pencil" || tool === "eraser") {
            overlayCanvas.style.cursor = "none";
        } else {
            overlayCanvas.style.cursor = tool === "move" ? "default" : "crosshair";
            drawCursor(); // clear dynamic cursor 
        }
    }

    btnMove.onclick = () => setActiveTool("move");
    btnPencil.onclick = () => setActiveTool("pencil");
    btnEraser.onclick = () => setActiveTool("eraser");
    btnText.onclick = () => setActiveTool("text");
    btnArrow.onclick = () => setActiveTool("arrow");

    // Setting Controls
    const labelSize = document.createElement("span"); labelSize.innerText = "Tamaño:"; labelSize.style.fontSize = "14px";
    const inputSize = document.createElement("input");
    inputSize.type = "range"; inputSize.min = "1"; inputSize.max = "300"; inputSize.value = brushSize;
    inputSize.style.width = "100px";
    inputSize.title = "Grosor y tamaño";

    const selectShape = document.createElement("select");
    const optRound = document.createElement("option"); optRound.value = "round"; optRound.innerHTML = "⚫";
    const optSquare = document.createElement("option"); optSquare.value = "square"; optSquare.innerHTML = "⬛";
    selectShape.appendChild(optRound); selectShape.appendChild(optSquare);
    selectShape.title = "Forma del pincel";

    // Actions
    const btnUndo = document.createElement("button"); btnUndo.innerHTML = `<img src="${getIconUrl('volver.png')}" style="width:20px;height:20px;filter:invert(1);">`; btnUndo.title = "Deshacer";
    const btnRedo = document.createElement("button"); btnRedo.innerHTML = `<img src="${getIconUrl('volver.png')}" style="width:20px;height:20px;transform:scaleX(-1);filter:invert(1);">`; btnRedo.title = "Rehacer";
    const btnDelete = document.createElement("button"); btnDelete.innerHTML = `<img src="${getIconUrl('eliminar.png')}" style="width:20px;height:20px;filter:invert(1);">`; btnDelete.title = "Eliminar Elemento Seleccionado (Del)";

    [btnUndo, btnRedo, btnDelete].forEach(b => Object.assign(b.style, { width: "36px", height: "36px", background: "transparent", border: "1px solid #444", borderRadius: "4px", cursor: "pointer", display: "flex", justifyContent: "center", alignItems: "center" }));

    function deleteActiveObject() {
        if (activeObject) {
            const index = objects.indexOf(activeObject);
            if (index > -1) {
                objects.splice(index, 1);
                activeObject = null;
                renderVectorCanvas();
                saveState();
            }
        }
    }
    btnDelete.onclick = deleteActiveObject;

    // Save/Cancel
    const btnSave = document.createElement("button");
    btnSave.innerHTML = `<img src="${getIconUrl('salvar.png')}" style="width:16px;height:16px;vertical-align:middle;filter:invert(1);margin-right:6px;">Guardar Scribble`;
    Object.assign(btnSave.style, { background: "#4CAF50", color: "white", padding: "8px 16px", cursor: "pointer", border: "none", borderRadius: "4px", fontWeight: "bold" });

    const btnCancel = document.createElement("button");
    btnCancel.innerHTML = `<img src="${getIconUrl('cancelar.png')}" style="width:16px;height:16px;vertical-align:middle;filter:invert(1);margin-right:6px;">Cancelar`;
    Object.assign(btnCancel.style, { background: "#F44336", color: "white", padding: "8px 16px", cursor: "pointer", border: "none", borderRadius: "4px", fontWeight: "bold" });

    toolbar.append(btnMove, btnPencil, btnEraser, btnText, btnArrow);
    toolbar.append(document.createElement("div"), labelSize, inputSize, selectShape, document.createElement("div"));
    toolbar.append(btnUndo, btnRedo, btnDelete);

    const rightControls = document.createElement("div");
    rightControls.style.marginLeft = "auto";
    rightControls.style.display = "flex";
    rightControls.style.gap = "10px";
    rightControls.append(btnSave, btnCancel);
    toolbar.append(rightControls);

    // --- State & Canvas Setup ---
    const canvasContainer = document.createElement("div");
    Object.assign(canvasContainer.style, { position: "relative", border: "1px solid #555", flexGrow: "1", overflow: "hidden", display: "flex", justifyContent: "center", alignItems: "center", backgroundColor: "#333" });

    const innerWrapper = document.createElement("div");
    Object.assign(innerWrapper.style, { position: "relative" });

    const bgCanvas = document.createElement("canvas");
    const rasterCanvas = document.createElement("canvas");
    const vectorCanvas = document.createElement("canvas");
    const overlayCanvas = document.createElement("canvas");

    const layerStyle = { position: "absolute", top: "0", left: "0", pointerEvents: "none" };
    Object.assign(bgCanvas.style, layerStyle);
    Object.assign(rasterCanvas.style, layerStyle);
    Object.assign(vectorCanvas.style, layerStyle);
    Object.assign(overlayCanvas.style, { position: "relative", zIndex: "4", cursor: "none", display: "block" });

    innerWrapper.appendChild(bgCanvas);
    innerWrapper.appendChild(rasterCanvas);
    innerWrapper.appendChild(vectorCanvas);
    innerWrapper.appendChild(overlayCanvas);

    canvasContainer.appendChild(innerWrapper);
    container.appendChild(toolbar);
    container.appendChild(canvasContainer);
    overlay.appendChild(container);
    document.body.appendChild(overlay);

    const rCtx = rasterCanvas.getContext("2d", { willReadFrequently: true });
    const vCtx = vectorCanvas.getContext("2d");
    const oCtx = overlayCanvas.getContext("2d");

    let canvasWidth = 0; let canvasHeight = 0;

    // SVG Arrow Generation logic
    const arrowImg = new Image();
    const redArrowCanvas = document.createElement("canvas");
    const arrCtx = redArrowCanvas.getContext("2d");
    let arrowLoaded = false;
    arrowImg.onload = () => {
        redArrowCanvas.width = arrowImg.width || 100;
        redArrowCanvas.height = arrowImg.height || 100;
        arrCtx.drawImage(arrowImg, 0, 0, redArrowCanvas.width, redArrowCanvas.height);
        arrCtx.globalCompositeOperation = 'source-in';
        arrCtx.fillStyle = primaryColor;
        arrCtx.fillRect(0, 0, redArrowCanvas.width, redArrowCanvas.height);
        arrowLoaded = true;
    };
    arrowImg.src = getIconUrl('arrow_right.svg');

    let isDrawingRaster = false;
    let isDraggingObject = false;
    let activeObject = null;
    let dragOffsetX = 0; let dragOffsetY = 0;
    let isCreatingArrow = false; let newArrowStart = { x: 0, y: 0 };

    let history = []; let historyStep = -1;
    let objects = [];

    inputSize.oninput = (e) => {
        brushSize = parseInt(e.target.value);
        drawCursor();

        // Dynamic Size Update
        if (activeTool === "move" && activeObject) {
            if (activeObject.type === 'text') {
                activeObject.size = brushSize * 4;
            } else if (activeObject.type === 'arrow') {
                activeObject.size = brushSize;
            }
            renderVectorCanvas(); // refresh instantly
        }
    };

    inputSize.onchange = () => {
        // Save state only when slider drag completes to avoid history flooding
        if (activeTool === "move" && activeObject) saveState();
    };

    selectShape.onchange = (e) => { brushShape = e.target.value; drawCursor(); };

    function saveState() {
        historyStep++;
        if (historyStep < history.length) { history.length = historyStep; }
        history.push({
            rasterData: rasterCanvas.toDataURL(),
            objectsData: JSON.parse(JSON.stringify(objects))
        });
    }

    function restoreState(index) {
        if (index >= 0 && index < history.length) {
            const snap = history[index];
            objects = JSON.parse(JSON.stringify(snap.objectsData));
            const img = new Image();
            img.onload = () => {
                rCtx.clearRect(0, 0, canvasWidth, canvasHeight);
                rCtx.drawImage(img, 0, 0);
                activeObject = null; // deselect
                renderVectorCanvas();
            };
            img.src = snap.rasterData;
        } else if (index === -1) {
            rCtx.clearRect(0, 0, canvasWidth, canvasHeight);
            objects = [];
            activeObject = null;
            renderVectorCanvas();
        }
    }

    btnUndo.onclick = () => { if (historyStep > -1) { historyStep--; restoreState(historyStep); } };
    btnRedo.onclick = () => { if (historyStep < history.length - 1) { historyStep++; restoreState(historyStep); } };

    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => {
        const maxWidth = window.innerWidth * 0.8; const maxHeight = window.innerHeight * 0.8;
        let w = img.width; let h = img.height;
        let displayW = w; let displayH = h;
        const ratio = Math.min(maxWidth / w, maxHeight / h);
        if (ratio < 1) { displayW *= ratio; displayH *= ratio; }

        canvasWidth = w; canvasHeight = h;
        [bgCanvas, rasterCanvas, vectorCanvas, overlayCanvas].forEach(c => {
            c.width = w; c.height = h;
            c.style.width = `${displayW}px`;
            c.style.height = `${displayH}px`;
        });

        const bgCtx = bgCanvas.getContext("2d");
        bgCtx.drawImage(img, 0, 0, w, h);

        if (dataWidget.value && dataWidget.value.startsWith("data:image")) {
            const existingLayer = new Image();
            existingLayer.onload = () => { rCtx.drawImage(existingLayer, 0, 0, w, h); saveState(); };
            existingLayer.src = dataWidget.value;
        } else { saveState(); }
    };

    let src_filename = imageName;
    let src_type = "input";
    let src_subfolder = "";

    const typeMatch = src_filename.match(/(.*)\s+\[(.*)\]/);
    if (typeMatch) {
        src_filename = typeMatch[1];
        src_type = typeMatch[2];
    }

    if (src_filename.includes("/")) {
        const parts = src_filename.split("/");
        src_filename = parts.pop();
        src_subfolder = parts.join("/");
        if (src_subfolder === "clipspace" && src_type === "input") {
            src_type = "temp";
        }
    }

    img.src = api.apiURL(`/view?filename=${encodeURIComponent(src_filename)}&type=${encodeURIComponent(src_type)}&subfolder=${encodeURIComponent(src_subfolder)}&t=${Date.now()}`);

    function renderVectorCanvas() {
        vCtx.clearRect(0, 0, canvasWidth, canvasHeight);
        vCtx.fillStyle = primaryColor; vCtx.strokeStyle = primaryColor;

        objects.forEach(obj => {
            if (obj.type === 'text') {
                vCtx.font = `${Math.floor(obj.size)}px 'Caveat', 'Comic Sans MS', cursive, sans-serif`;
                vCtx.textBaseline = "top";
                vCtx.fillText(obj.text, obj.x, obj.y);
            } else if (obj.type === 'arrow' && arrowLoaded) {
                const dx = obj.endX - obj.x; const dy = obj.endY - obj.y;
                const angle = Math.atan2(dy, dx);
                const dist = Math.hypot(dx, dy);
                vCtx.save();
                vCtx.translate(obj.x, obj.y);
                vCtx.rotate(angle);
                const thickness = obj.size * 2;
                vCtx.drawImage(redArrowCanvas, 0, -thickness / 2, dist, thickness);
                vCtx.restore();
            }
        });

        if (activeTool === "move" && activeObject) {
            vCtx.save();
            vCtx.strokeStyle = 'cyan'; vCtx.lineWidth = 2; vCtx.setLineDash([5, 5]);
            if (activeObject.type === 'text') {
                vCtx.font = `${Math.floor(activeObject.size)}px 'Caveat', 'Comic Sans MS', cursive, sans-serif`;
                const width = vCtx.measureText(activeObject.text).width;
                vCtx.strokeRect(activeObject.x - 5, activeObject.y - 5, width + 10, activeObject.size + 10);
            } else if (activeObject.type === 'arrow') {
                const dx = activeObject.endX - activeObject.x; const dy = activeObject.endY - activeObject.y;
                const angle = Math.atan2(dy, dx); const dist = Math.hypot(dx, dy);
                vCtx.translate(activeObject.x, activeObject.y);
                vCtx.rotate(angle);
                vCtx.strokeRect(0, -(activeObject.size), dist, activeObject.size * 2);
            }
            vCtx.restore();
        }
    }

    function hitTestObject(x, y) {
        for (let i = objects.length - 1; i >= 0; i--) {
            const obj = objects[i];
            if (obj.type === 'text') {
                vCtx.font = `${Math.floor(obj.size)}px 'Caveat', 'Comic Sans MS', cursive, sans-serif`;
                const width = vCtx.measureText(obj.text).width;
                if (x >= obj.x && x <= obj.x + width && y >= obj.y && y <= obj.y + obj.size) return obj;
            } else if (obj.type === 'arrow') {
                const A = x - obj.x; const B = y - obj.y;
                const C = obj.endX - obj.x; const D = obj.endY - obj.y;
                const dot = A * C + B * D; const len_sq = C * C + D * D;
                let param = -1; if (len_sq !== 0) param = dot / len_sq;
                let xx, yy;
                if (param < 0) { xx = obj.x; yy = obj.y; }
                else if (param > 1) { xx = obj.endX; yy = obj.endY; }
                else { xx = obj.x + param * C; yy = obj.y + param * D; }
                const dx = x - xx; const dy = y - yy;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist <= obj.size * 2) return obj;
            }
        }
        return null;
    }

    function getMousePos(evt) {
        const rect = overlayCanvas.getBoundingClientRect();
        return {
            x: (evt.clientX - rect.left) * (overlayCanvas.width / rect.width),
            y: (evt.clientY - rect.top) * (overlayCanvas.height / rect.height)
        };
    }

    let mousePos = { x: 0, y: 0 };

    function drawCursor() {
        oCtx.clearRect(0, 0, canvasWidth, canvasHeight);
        if (activeTool === "pencil" || activeTool === "eraser") {
            oCtx.beginPath();
            if (brushShape === "round") { oCtx.arc(mousePos.x, mousePos.y, brushSize / 2, 0, Math.PI * 2); }
            else { oCtx.rect(mousePos.x - brushSize / 2, mousePos.y - brushSize / 2, brushSize, brushSize); }
            oCtx.lineWidth = 1;
            oCtx.strokeStyle = "rgba(255,255,255,0.8)"; oCtx.stroke();
            oCtx.strokeStyle = "rgba(0,0,0,0.5)"; oCtx.stroke();
        }
    }

    overlayCanvas.onmousemove = (e) => {
        mousePos = getMousePos(e); drawCursor();

        if (isDrawingRaster) {
            const ctx = rCtx;
            ctx.lineCap = brushShape; ctx.lineJoin = brushShape; ctx.lineWidth = brushSize;
            if (activeTool === "eraser") { ctx.globalCompositeOperation = "destination-out"; ctx.strokeStyle = "rgba(0,0,0,1)"; }
            else { ctx.globalCompositeOperation = "source-over"; ctx.strokeStyle = primaryColor; }
            ctx.lineTo(mousePos.x, mousePos.y);
            ctx.stroke();
        }

        if (isDraggingObject && activeObject) {
            if (activeObject.type === "text") {
                activeObject.x = mousePos.x - dragOffsetX;
                activeObject.y = mousePos.y - dragOffsetY;
            } else if (activeObject.type === "arrow") {
                if (dragOffsetX === "start") { activeObject.x = mousePos.x; activeObject.y = mousePos.y; }
                else if (dragOffsetX === "end") { activeObject.endX = mousePos.x; activeObject.endY = mousePos.y; }
                else {
                    const pdx = activeObject.endX - activeObject.x; const pdy = activeObject.endY - activeObject.y;
                    activeObject.x = mousePos.x - dragOffsetX; activeObject.y = mousePos.y - dragOffsetY;
                    activeObject.endX = activeObject.x + pdx; activeObject.endY = activeObject.y + pdy;
                }
            }
            renderVectorCanvas();
        }

        if (isCreatingArrow && arrowLoaded) {
            oCtx.save();
            const dx = mousePos.x - newArrowStart.x; const dy = mousePos.y - newArrowStart.y;
            const angle = Math.atan2(dy, dx); const dist = Math.hypot(dx, dy);
            oCtx.translate(newArrowStart.x, newArrowStart.y);
            oCtx.rotate(angle);
            const thickness = brushSize * 2;
            oCtx.drawImage(redArrowCanvas, 0, -thickness / 2, dist, thickness);
            oCtx.restore();
        }
    };

    overlayCanvas.onmousedown = (e) => {
        mousePos = getMousePos(e);
        if (activeTool === "move") {
            const hit = hitTestObject(mousePos.x, mousePos.y);
            if (hit) {
                activeObject = hit; isDraggingObject = true;

                // Set sizeslider to object size
                if (hit.type === 'text') { brushSize = Math.max(1, hit.size / 4); }
                else if (hit.type === 'arrow') { brushSize = hit.size; }
                inputSize.value = brushSize;
                selectShape.disabled = true; // irrelevant while moving object

                if (hit.type === 'arrow') {
                    const dStart = Math.hypot(hit.x - mousePos.x, hit.y - mousePos.y);
                    const dEnd = Math.hypot(hit.endX - mousePos.x, hit.endY - mousePos.y);
                    if (dStart < hit.size * 2) { dragOffsetX = "start"; }
                    else if (dEnd < hit.size * 2) { dragOffsetX = "end"; }
                    else { dragOffsetX = mousePos.x - hit.x; dragOffsetY = mousePos.y - hit.y; }
                } else {
                    dragOffsetX = mousePos.x - hit.x; dragOffsetY = mousePos.y - hit.y;
                }
                objects.splice(objects.indexOf(hit), 1); objects.push(hit);
                renderVectorCanvas();
            } else {
                activeObject = null;
                selectShape.disabled = false;
                renderVectorCanvas();
            }
        }
        else if (activeTool === "pencil" || activeTool === "eraser") {
            activeObject = null; selectShape.disabled = false; renderVectorCanvas();
            isDrawingRaster = true; rCtx.beginPath(); rCtx.moveTo(mousePos.x, mousePos.y);
        }
        else if (activeTool === "arrow") {
            activeObject = null; selectShape.disabled = false; renderVectorCanvas();
            isCreatingArrow = true; newArrowStart = { x: mousePos.x, y: mousePos.y };
        }
    };

    overlayCanvas.onmouseup = (e) => {
        if (isDrawingRaster) { isDrawingRaster = false; rCtx.closePath(); saveState(); }
        if (isDraggingObject) { isDraggingObject = false; saveState(); }
        if (isCreatingArrow) {
            isCreatingArrow = false; mousePos = getMousePos(e);
            if (Math.hypot(mousePos.x - newArrowStart.x, mousePos.y - newArrowStart.y) > 5) {
                objects.push({ type: 'arrow', x: newArrowStart.x, y: newArrowStart.y, endX: mousePos.x, endY: mousePos.y, size: brushSize });
                renderVectorCanvas(); saveState();
            }
            drawCursor();
        }
    };

    overlayCanvas.onclick = (e) => {
        if (activeTool === "text") {
            activeObject = null; selectShape.disabled = false; renderVectorCanvas();
            const pos = getMousePos(e);
            let userText = prompt("Introduce el texto (Rojo Handwriting):", "");
            if (userText && userText.trim() !== "") {
                objects.push({ type: 'text', text: userText, x: pos.x, y: pos.y, size: brushSize * 4 });
                renderVectorCanvas(); saveState();
                setActiveTool("move");
            }
        }
    };

    overlayCanvas.ondblclick = (e) => {
        if (activeTool === "move") {
            const hit = hitTestObject(mousePos.x, mousePos.y);
            if (hit && hit.type === 'text') {
                let userText = prompt("Editar texto:", hit.text);
                if (userText !== null) { hit.text = userText; renderVectorCanvas(); saveState(); }
            }
        }
    }

    // Keyboard support for deleting selected element
    const handleKeyDown = (e) => {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            // Prevent deleting if the user is typing in a text prompt/input (if any active element is an input)
            if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                deleteActiveObject();
            }
        }
    };
    window.addEventListener("keydown", handleKeyDown);

    function closeEditor() {
        window.removeEventListener("keydown", handleKeyDown);
        document.body.removeChild(overlay);
    }

    btnSave.onclick = () => {
        // Deselect object so selection bounding box isn't rendered into the saved scribble image
        activeObject = null;
        renderVectorCanvas();

        const finalCanvas = document.createElement("canvas");
        finalCanvas.width = canvasWidth; finalCanvas.height = canvasHeight;
        const ctx = finalCanvas.getContext("2d");
        ctx.drawImage(rasterCanvas, 0, 0); ctx.drawImage(vectorCanvas, 0, 0);
        dataWidget.value = finalCanvas.toDataURL("image/png");
        closeEditor();
        app.graph.setDirtyCanvas(true);
    };

    btnCancel.onclick = () => { closeEditor(); };
}

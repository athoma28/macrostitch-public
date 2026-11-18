"""Dependency-free local mask editor served by the workspace API."""

EDITOR_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Macrostitch · Mask editor</title>
  <style>
    [hidden] { display: none !important; }
    :root { color-scheme: dark; font: 15px system-ui; background: #101719; color: #e4ece8; }
    body { margin: 0; height: 100dvh; display: grid; grid-template-rows: auto minmax(0, 1fr); overflow: hidden; }
    header { padding: 24px 32px; border-bottom: 1px solid #33423d; }
    #saveState { display: inline-block; margin: 8px 0 0; color: #b9e9c6; }
    .mobile-nav { display: none; }
    h1 { font-size: 24px; margin: 0 0 6px; }
    p { line-height: 1.5; color: #abbfb6; }
    main { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 24px; padding: 24px 32px; min-height: 0; overflow: hidden; }
    aside, article { min-width: 0; min-height: 0; overflow: auto; scrollbar-gutter: stable; }
    aside { padding-right: 6px; }
    article { padding-right: 6px; }
    section { background: #182321; border: 1px solid #33423d; border-radius: 12px; padding: 18px; margin-bottom: 16px; }
    h2 { font-size: 17px; margin: 0 0 16px; }
    label { display: block; margin: 12px 0 6px; }
    input, textarea, select, button {
      box-sizing: border-box; max-width: 100%; font: inherit; border: 1px solid #536a60;
      border-radius: 6px; background: #101917; color: inherit; padding: 9px;
    }
    input:not([type=checkbox]), textarea, select { width: 100%; }
    input[type=checkbox] { width: auto; }
    button { cursor: pointer; margin: 6px 4px 0 0; }
    button:hover { background: #344b41; }
    button[aria-pressed=true] { border-color: #b9e9c6; background: #344b41; }
    button:disabled { opacity: .4; cursor: default; }
    input[aria-invalid=true], textarea[aria-invalid=true] { border-color: #ffb4a6; }
    :focus-visible { outline: 3px solid #b9e9c6; outline-offset: 3px; }
    .primary { background: #b9e9c6; color: #122419; font-weight: 650; }
    .primary:hover { background: #d0f5d9; }
    small { display: block; color: #abbfb6; line-height: 1.5; margin-top: 6px; }
    .stage { height: max(180px, calc(100dvh - 430px)); min-height: 180px; max-height: max(180px, calc(100dvh - 430px)); overflow: auto; background: repeating-conic-gradient(#24332e 0% 25%, #1b2823 0% 50%) 50%/24px 24px; }
    canvas { display: block; margin-inline: auto; width: 100%; max-width: none; height: auto; touch-action: none; cursor: crosshair; }
    #empty { padding: 40px; }
    #status, #feedback { white-space: pre-wrap; overflow-wrap: anywhere; }
    #feedback { margin-top: 12px; }
    .canvas-tools { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: center; margin-bottom: 10px; }
    .canvas-tools button { min-height: 42px; }
    #canvasFeedback, #renderFeedback { white-space: pre-wrap; overflow-wrap: anywhere; margin-top: 10px; }
    #canvasFeedback:empty, #renderFeedback:empty { display: none; }
    #canvasFeedback { color: #abbfb6; }
    #canvasFeedback.error { color: #ffb4a6; }
    #masks { padding-left: 20px; }
    li { margin: 9px 0; overflow-wrap: anywhere; }
    a { color: #b9e9c6; }
    #result { max-width: 100%; height: auto; background: repeating-conic-gradient(#24332e 0% 25%, #1b2823 0% 50%) 50%/24px 24px; }
    output { float: right; }
    #links a { display: inline-block; margin: 8px 14px 0 0; }
    #links .download { border: 1px solid #536a60; border-radius: 6px; padding: 10px 14px; text-decoration: none; }
    #artifactLinks a { display: inline-block; margin: 8px 14px 0 0; overflow-wrap: anywhere; }
    #artifactDetails { margin: 16px 0; }
    #resultState { border-left: 3px solid #b9e9c6; padding: 8px 12px; margin: 16px 0 8px; }
    #resultState.stale { border-color: #f3c87d; color: #f3c87d; }
    progress { width: 100%; margin-top: 14px; accent-color: #b9e9c6; }
    .error { color: #ffb4a6; }
    @media(max-width: 800px) {
      body { display: block; height: auto; overflow: visible; }
      main { grid-template-columns: 1fr; padding: 16px; overflow: visible; }
      aside, article { overflow: visible; padding-right: 0; scrollbar-gutter: auto; }
      header { padding: 20px; }
      .mobile-nav { display: flex; position: sticky; top: 0; z-index: 2; gap: 6px; padding: 8px 16px; background: #101719; border-bottom: 1px solid #33423d; }
      .mobile-nav button { margin: 0; flex: 1; min-height: 42px; padding: 8px 4px; }
      section, h2 { scroll-margin-top: 74px; }
      .stage { height: clamp(220px, 35vh, 320px); min-height: 220px; max-height: 320px; }
    }
  </style>
</head>
<body>
<header><h1>Macrostitch mask editor</h1><span>Choose what to stack, preserve, or remove.</span><br><small id="saveState" role="status">No unsaved recipe changes</small></header>
<nav class="mobile-nav" aria-label="Editor sections">
  <button id="jumpSetup">Setup</button><button id="jumpMasks">Masks</button>
  <button id="jumpCanvas">Canvas</button><button id="jumpResults">Results</button>
</nav>
<main>
  <aside tabindex="0" aria-label="Mask editor controls">
    <section id="setupSection" tabindex="-1">
      <h2>1. Load your work</h2>
      <label for="recipe">Recipe JSON</label>
      <input id="recipe" type="file" accept=".json,application/json">
      <small>All imported recipe settings are retained. Use the configuration JSON, not a run report.</small>
      <label for="photo">Reference image</label>
      <input id="photo" type="file" accept="image/*">
      <small>Use reference.png from a run for exact mask placement. Other photos must match the reference frame and aspect ratio. Coordinates use the full reference canvas.</small>
      <div id="recipeInfo"></div>
      <details>
        <summary>Browse workspace videos and recipes</summary>
        <div id="workspacePath"></div>
        <button id="workspaceUp" disabled>Parent folder</button>
        <button id="workspaceRefresh">Refresh files</button>
        <div id="workspaceStatus" role="status"></div>
        <ul id="workspaceFiles" aria-label="Workspace files"></ul>
      </details>
    </section>
    <section>
      <h2>2. Background</h2>
      <label><input id="override" type="checkbox"> Override recipe background opacity</label>
      <label for="opacity">Background retained <output id="opacityValue">100%</output></label>
      <input id="opacity" type="range" min="0" max="1" step=".01" value="1" disabled>
      <small>0% transparent · 100% reference image. Excluded objects stay transparent.</small>
      <label for="margin">Surrounding context (reference pixels)</label>
      <input id="margin" type="number" min="0" max="512" step="1" value="0">
      <small id="marginError" class="error" hidden></small>
      <small>Expands retained subject context. Final feathering and alignment appear after rendering.</small>
    </section>
    <section>
      <h2 id="maskHeading" tabindex="-1">3. Object masks</h2>
      <div id="editMode" role="status">Drawing a new mask</div>
      <label for="maskName">Object name</label>
      <input id="maskName" maxlength="80" placeholder="e.g. foreground leaf">
      <label for="action">Treatment</label>
      <select id="action">
        <option value="stack">Stack — include in focused subject</option>
        <option value="reference">Reference — preserve original object</option>
        <option value="exclude">Exclude — make object transparent</option>
      </select>
      <small id="drawingHelp">Click to add polygon points; drag a point to adjust it. Choose Edit on a saved mask to change its name, treatment, or shape. At overlaps: exclude wins, then reference, then stack.</small>
      <button id="add">Add polygon</button>
      <button id="undo">Undo point</button>
      <button id="cancel">Cancel drawing</button>
      <button id="goCanvas">Go to canvas</button>
      <div id="pointCount" aria-live="polite">0 points</div>
      <div role="group" aria-label="Saved mask history">
        <button id="undoMask">Undo mask change</button>
        <button id="redoMask">Redo mask change</button>
      </div>
      <small>Outside text fields: Enter saves, Escape cancels, Backspace removes the last point. Ctrl/⌘ Z undoes a saved mask change; Ctrl/⌘ Shift Z redoes it.</small>
      <div id="feedback" role="status" aria-live="polite"></div>
      <ul id="masks" aria-label="Object masks"></ul>
      <div id="subjectInfo"></div>
    </section>
    <section>
      <h2>4. Render</h2>
      <label for="inputs">Video paths (one per line)</label>
      <textarea id="inputs" rows="3" placeholder="videos/clip.mov"></textarea>
      <small id="inputsError" class="error" hidden></small>
      <label for="destination">Output directory</label>
      <input id="destination" value="outputs/masked">
      <small id="destinationError" class="error" hidden></small>
      <small>Paths are relative to the server workspace. Choose a dedicated output directory.</small>
      <button id="download">Download recipe</button>
      <button id="run" class="primary">Render stack</button>
      <div id="renderFeedback"></div>
    </section>
  </aside>
  <article tabindex="0" aria-label="Reference and render results">
    <section>
      <h2 id="canvasHeading" tabindex="-1">Reference canvas · mask overlay</h2>
      <p>This is a placement preview. Background opacity, surrounding context, and the final stack are applied when you render. Imported subject polygons are outlined in white; stack masks green, reference blue, exclude red.</p>
      <div class="canvas-tools">
      <div role="group" aria-label="Canvas tool">
        <button id="drawTool" aria-pressed="true">Draw / edit</button>
        <button id="panTool" aria-pressed="false">Pan</button>
      </div>
      <div role="group" aria-label="Reference zoom">
        <button id="fit" aria-pressed="true">Fit</button>
        <button id="zoom2" aria-pressed="false">2×</button>
        <button id="zoom4" aria-pressed="false">4×</button>
      </div>
      <label><input id="showMasks" type="checkbox" checked> Show saved masks</label>
      </div>
      <small id="canvasHelp">Draw / edit adds or moves points. Pan drags the image without changing masks.</small>
      <div class="stage" id="stage">
        <div id="empty">Load a reference image to draw masks.</div>
        <canvas id="canvas" hidden tabindex="0" aria-label="Reference image with object mask polygons" aria-describedby="drawingHelp"></canvas>
      </div>
      <div class="canvas-tools" role="group" aria-label="Current polygon">
        <span id="canvasPointCount">0 points</span>
        <button id="canvasSave" disabled>Save polygon</button>
        <button id="canvasUndo" disabled>Undo point</button>
        <button id="canvasCancel" disabled>Cancel</button>
        <button id="maskSettings">Name &amp; treatment</button>
      </div>
      <div id="canvasFeedback"></div>
    </section>
    <section>
      <h2>Recent jobs</h2>
      <button id="refreshJobs">Refresh jobs</button>
      <div id="recentStatus" role="status"></div>
      <ul id="recentJobs" aria-label="Recent render jobs"></ul>
    </section>
    <section>
      <h2 id="resultsHeading" tabindex="-1">Job status</h2>
      <div id="status" role="status" aria-live="polite">Ready. Load a recipe to preserve your existing alignment and focus settings.</div>
      <progress id="jobProgress" aria-label="Current rendering stage progress" hidden></progress>
      <div id="links"></div>
      <details id="artifactDetails" hidden>
        <summary>Reference, detail crops &amp; validation files</summary>
        <div id="artifactLinks"></div>
      </details>
      <p id="resultState" hidden></p>
      <div id="resultViews" role="group" aria-label="Result preview" hidden></div>
      <p id="resultCaption" hidden></p>
      <img id="result" alt="Completed stack" aria-describedby="resultCaption" hidden>
    </section>
  </article>
</main>
<script>
'use strict';
const $ = id => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
for (const [button, target] of [['jumpSetup', 'setupSection'], ['jumpMasks', 'maskHeading'],
  ['jumpCanvas', 'canvasHeading'], ['jumpResults', 'resultsHeading']]) {
  $(button).onclick = () => { $(target).focus(); $(target).scrollIntoView({block: 'start'}); };
}
const colors = {stack: '#7ded9d', reference: '#78b8ff', exclude: '#ff887b'};
let recipe = {}, masks = [], points = [], photo = null, activeJob = null, zoom = 1;
let editingIndex = null, draggedPoint = null;
let undoMasks = [], redoMasks = [];
let savedConfig = null;
let recipeLoadEpoch = 0, photoLoadEpoch = 0;
let canvasTool = 'draw', panStart = null;
let dragPointer = null;
let activeRenderState = null, renderedState = null;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
  $('renderFeedback').textContent = message;
  $('renderFeedback').classList.toggle('error', error);
}

function feedback(message, error = false) {
  $('feedback').textContent = message;
  $('feedback').classList.toggle('error', error);
  $('canvasFeedback').textContent = message;
  $('canvasFeedback').classList.toggle('error', error);
}

function updateSaveState() {
  const message = hasDraft() ? 'Polygon in progress · save or cancel to continue'
    : JSON.stringify(config()) !== savedConfig ? 'Unsaved recipe changes · download or render to save'
    : 'No unsaved recipe changes';
  if ($('saveState').textContent !== message) $('saveState').textContent = message;
  updateResultState();
}

function renderState(configuration = config(), inputs = videoInputs()) {
  return JSON.stringify({configuration, inputs});
}
function videoInputs() { return $('inputs').value.split(/\r?\n/).map(value => value.trim()).filter(Boolean); }
function updateResultState() {
  $('resultState').hidden = renderedState === null || $('result').hidden;
  if ($('resultState').hidden) return;
  const stale = hasDraft() || renderState() !== renderedState;
  $('resultState').textContent = stale
    ? 'Preview is from the last render. Render again to include your changes.'
    : 'Preview matches the current recipe and videos.';
  $('resultState').classList.toggle('stale', stale);
}

function config() {
  const result = structuredClone(recipe);
  result.background_margin_px = Number($('margin').value);
  if ($('override').checked) result.background_opacity = Number($('opacity').value);
  else delete result.background_opacity;
  if (masks.length) result.object_masks = structuredClone(masks);
  else delete result.object_masks;
  return result;
}

function drawPoly(poly, color, fill = true, closed = fill) {
  if (!poly.length) return;
  ctx.beginPath();
  poly.forEach(([x, y], i) => i ? ctx.lineTo(x * canvas.width, y * canvas.height)
                               : ctx.moveTo(x * canvas.width, y * canvas.height));
  if (closed) ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, canvas.width / 500);
  if (fill) { ctx.fillStyle = color + '30'; ctx.fill(); }
  ctx.stroke();
  for (const [x, y] of poly) {
    ctx.beginPath();
    ctx.arc(x * canvas.width, y * canvas.height, Math.max(3, canvas.width / 300), 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

function hasDraft() { return editingIndex !== null || points.length > 0; }

function draw() {
  if (photo) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(photo, 0, 0);
    if ($('showMasks').checked) {
      if (recipe.subject_polygon) drawPoly(recipe.subject_polygon, '#ffffff', false, true);
      masks.forEach((mask, i) => { if (i !== editingIndex) drawPoly(mask.polygon, colors[mask.action]); });
    }
    drawPoly(points, colors[$('action').value], false, editingIndex !== null);
  }
  $('pointCount').textContent = points.length + ' points' + (points.length >= 128 ? ' · maximum reached' : '');
  $('add').textContent = editingIndex === null ? 'Add polygon' : 'Save mask changes';
  $('add').disabled = points.length < 3 || (editingIndex === null && masks.length >= 32);
  $('undo').disabled = !points.length;
  $('cancel').disabled = !hasDraft();
  $('cancel').textContent = editingIndex === null ? 'Cancel drawing' : 'Cancel changes';
  $('undoMask').disabled = !undoMasks.length || hasDraft();
  $('redoMask').disabled = !redoMasks.length || hasDraft();
  $('editMode').textContent = editingIndex === null ? 'Drawing a new mask' : 'Editing ' + masks[editingIndex].name;
  $('canvasPointCount').textContent = $('pointCount').textContent;
  $('canvasSave').disabled = $('add').disabled;
  $('canvasSave').textContent = editingIndex === null ? 'Save polygon' : 'Save changes';
  $('canvasUndo').disabled = $('undo').disabled;
  $('canvasCancel').disabled = $('cancel').disabled;
  $('goCanvas').disabled = !photo;
  updateSaveState();
}

function resetDraft() {
  points = [];
  editingIndex = null;
  draggedPoint = null;
  dragPointer = null;
  $('maskName').value = '';
}

function rememberMasks() {
  undoMasks.push(structuredClone(masks));
  if (undoMasks.length > 100) undoMasks.shift();
  redoMasks = [];
}

function changeHistory(redo = false) {
  if (hasDraft()) { feedback('Save or cancel the current polygon before changing saved mask history.', true); return; }
  const source = redo ? redoMasks : undoMasks, target = redo ? undoMasks : redoMasks;
  if (!source.length) return;
  const control = document.activeElement?.id;
  target.push(structuredClone(masks));
  masks = source.pop();
  listMasks();
  feedback(redo ? 'Mask change redone.' : 'Mask change undone.');
  if (['undoMask', 'redoMask'].includes(control)) {
    ($(control).disabled ? $(redo ? 'undoMask' : 'redoMask') : $(control)).focus({preventScroll: true});
  }
}

function editMask(index) {
  if (hasDraft()) { feedback('Save or cancel the current polygon before editing another mask.', true); return; }
  editingIndex = index;
  setCanvasTool('draw');
  const mask = masks[index];
  points = structuredClone(mask.polygon);
  $('maskName').value = mask.name;
  $('action').value = mask.action;
  listMasks();
  feedback('Editing ' + mask.name + '. Drag points or change the name and treatment, then save.');
  $('maskName').focus();
}

function removeMask(index) {
  if (hasDraft()) { feedback('Save or cancel the current polygon before removing a mask.', true); return; }
  const name = masks[index].name;
  rememberMasks();
  masks.splice(index, 1);
  listMasks();
  feedback(name + ' removed. Undo mask change restores it.');
  const nextRow = $('masks').children[Math.min(index, masks.length - 1)];
  (nextRow ? nextRow.children[1] : $('maskName')).focus();
}

function listMasks() {
  const list = $('masks');
  list.replaceChildren();
  masks.forEach((mask, i) => {
    const li = document.createElement('li'), label = document.createElement('span');
    label.textContent = mask.name + ' · ' + mask.action;
    label.style.color = colors[mask.action];
    li.append(label);
    for (const [title, action] of [['Edit', editMask], ['Remove', removeMask]]) {
      const button = document.createElement('button');
      button.textContent = title;
      button.setAttribute('aria-label', title + ' ' + mask.name);
      button.disabled = hasDraft();
      button.onclick = () => action(i);
      li.append(button);
    }
    list.append(li);
  });
  $('subjectInfo').textContent = recipe.subject_polygon
    ? 'Imported subject polygon retained (' + recipe.subject_polygon.length + ' points).'
    : 'No imported subject polygon; stack masks or the recipe ROI define the subject.';
  draw();
}

function validPolygon(poly) {
  return Array.isArray(poly) && poly.length >= 3 && poly.length <= 128 && poly.every(p =>
    Array.isArray(p) && p.length === 2 && p.every(n =>
      typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1));
}

// Match the server's simple-polygon rules before the user submits a render.
function polygonError(poly) {
  if (!validPolygon(poly)) return 'Use 3–128 points within the reference image.';
  if (new Set(poly.map(point => point.join(','))).size !== poly.length)
    return 'Polygon points must be distinct. Do not repeat the closing point.';
  const edges = poly.map((point, i) => [point, poly[(i + 1) % poly.length]]);
  const area = Math.abs(edges.reduce((sum, [a, b]) => sum + a[0] * b[1] - a[1] * b[0], 0)) / 2;
  if (area < 1e-10) return 'The polygon must cover an area. Move points away from a straight line.';
  const orientation = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const touches = (a, b, p) => Math.abs(orientation(a, b, p)) < 1e-12 &&
    Math.min(a[0], b[0]) - 1e-12 <= p[0] && p[0] <= Math.max(a[0], b[0]) + 1e-12 &&
    Math.min(a[1], b[1]) - 1e-12 <= p[1] && p[1] <= Math.max(a[1], b[1]) + 1e-12;
  for (let i = 0; i < edges.length; i++) {
    for (let j = i + 1; j < edges.length; j++) {
      if (j === i + 1 || (i === 0 && j === edges.length - 1)) continue;
      const [a, b] = edges[i], [c, d] = edges[j];
      const crossing = orientation(a, b, c) * orientation(a, b, d) < 0 &&
                       orientation(c, d, a) * orientation(c, d, b) < 0;
      if (crossing || touches(a, b, c) || touches(a, b, d) || touches(c, d, a) || touches(c, d, b))
        return 'Polygon edges must not cross or touch. Drag the crossing points to fix the shape.';
    }
  }
  return null;
}

function applyRecipe(value, label) {
  if (!value || Array.isArray(value) || typeof value !== 'object' || value.config || value.artifacts)
    throw Error('Load a configuration object, not a job or run report.');
  if (value.subject_polygon != null && !validPolygon(value.subject_polygon)) throw Error('Invalid subject polygon');
  const next = value.object_masks ?? [];
  if (!Array.isArray(next) || next.length > 32 || next.some(mask =>
      !mask || typeof mask.name !== 'string' || !mask.name.trim() || mask.name.trim().length > 80 ||
      !Object.hasOwn(colors, mask.action) || polygonError(mask.polygon)) ||
      new Set(next.map(mask => mask.name.trim())).size !== next.length) throw Error('Invalid object masks');
  if (value.background_opacity != null && !(typeof value.background_opacity === 'number' &&
      value.background_opacity >= 0 && value.background_opacity <= 1)) throw Error('Invalid background opacity');
  if (value.background_margin_px != null && !(typeof value.background_margin_px === 'number' &&
      value.background_margin_px >= 0 && value.background_margin_px <= 512)) throw Error('Invalid background margin');
  recipe = structuredClone(value);
  masks = structuredClone(next);
  undoMasks = [];
  redoMasks = [];
  resetDraft();
  $('override').checked = value.background_opacity != null;
  $('opacity').value = value.background_opacity ?? (value.background_mode === 'transparent' ? 0 : 1);
  $('margin').value = value.background_margin_px ?? 0;
  clearFieldError('margin');
  syncOpacity();
  $('recipeInfo').textContent = 'Loaded ' + label + ' · ' + Object.keys(recipe).length + ' settings retained';
  feedback('Recipe loaded. Existing mask history has been reset.');
  listMasks();
  savedConfig = JSON.stringify(config());
  updateSaveState();
}

function canReplaceRecipe() {
  if (hasDraft()) { feedback('Save or cancel the current polygon before loading a recipe.', true); return false; }
  return JSON.stringify(config()) === savedConfig || window.confirm('Discard unsaved recipe changes and load another recipe?');
}

$('recipe').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  const request = ++recipeLoadEpoch;
  try {
    const value = JSON.parse(await file.text());
    if (request !== recipeLoadEpoch || !canReplaceRecipe()) return;
    applyRecipe(value, file.name);
    status('Recipe loaded. Other settings will be preserved on download and render.');
  } catch (error) { if (request === recipeLoadEpoch) feedback(error.message, true); }
  finally { if (event.target.files[0] === file) event.target.value = ''; }
};

function fitCanvas() {
  if (!photo || !canvas.width || !canvas.height) return;
  // The stage has an explicit height. Observe its border box so canvas changes
  // and scrollbars cannot feed back into the measured available space.
  const bounds = $('stage').getBoundingClientRect();
  const scale = Math.min(bounds.width / canvas.width, bounds.height / canvas.height);
  if (scale > 0) canvas.style.width = (canvas.width * scale * zoom) + 'px';
  if (zoom === 1) { $('stage').scrollLeft = 0; $('stage').scrollTop = 0; }
}
function setZoom(value) {
  const stage = $('stage'), before = canvas.getBoundingClientRect();
  const bounds = stage.getBoundingClientRect();
  const width = stage.clientWidth ?? bounds.width, height = stage.clientHeight ?? bounds.height;
  const centerX = bounds.left + (stage.clientLeft ?? 0) + width / 2;
  const centerY = bounds.top + (stage.clientTop ?? 0) + height / 2;
  // A fitted dimension has no scroll position: start zooming from its middle.
  // Bounding rectangles account for the canvas's centered horizontal margins.
  const x = before.width > width ? Math.max(0, Math.min(1, (centerX - before.left) / before.width)) : .5;
  const y = before.height > height ? Math.max(0, Math.min(1, (centerY - before.top) / before.height)) : .5;
  zoom = value;
  fitCanvas();
  if (photo && zoom !== 1) {
    const after = canvas.getBoundingClientRect();
    stage.scrollLeft = Math.max(0, Math.min(Math.max(0, after.width - width),
      (stage.scrollLeft || 0) + after.left + x * after.width - centerX));
    stage.scrollTop = Math.max(0, Math.min(Math.max(0, after.height - height),
      (stage.scrollTop || 0) + after.top + y * after.height - centerY));
  }
  for (const [id, scale] of [['fit', 1], ['zoom2', 2], ['zoom4', 4]])
    $(id).setAttribute('aria-pressed', String(zoom === scale));
  if (zoom === 1) { stage.scrollLeft = 0; stage.scrollTop = 0; }
}
$('fit').onclick = () => setZoom(1);
$('zoom2').onclick = () => setZoom(2);
$('zoom4').onclick = () => setZoom(4);
if ('ResizeObserver' in window) {
  const stageObserver = new ResizeObserver(fitCanvas);
  stageObserver.observe($('stage'), {box: 'border-box'});
} else window.addEventListener('resize', fitCanvas);

function loadPhoto(url, revoke = false) {
  const request = ++photoLoadEpoch, image = new Image();
  image.onload = () => {
    if (revoke) URL.revokeObjectURL(url);
    if (request !== photoLoadEpoch) return;
    photo = image;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    setZoom(1);
    canvas.hidden = false;
    $('empty').hidden = true;
    draw();
  };
  image.onerror = () => {
    if (revoke) URL.revokeObjectURL(url);
    if (request === photoLoadEpoch) feedback('Could not load the reference image.', true);
  };
  image.src = url;
}
$('photo').onchange = event => {
  if (event.target.files[0]) loadPhoto(URL.createObjectURL(event.target.files[0]), true);
};

function setCanvasTool(tool) {
  canvasTool = tool;
  draggedPoint = null;
  dragPointer = null;
  panStart = null;
  $('drawTool').setAttribute('aria-pressed', String(tool === 'draw'));
  $('panTool').setAttribute('aria-pressed', String(tool === 'pan'));
  canvas.style.cursor = tool === 'pan' ? 'grab' : 'crosshair';
  $('canvasHelp').textContent = tool === 'pan'
    ? 'Pan: drag to move around the image. Switch to Draw / edit to change points.'
    : 'Draw / edit adds or moves points. Pan drags the image without changing masks.';
}
$('drawTool').onclick = () => setCanvasTool('draw');
$('panTool').onclick = () => setCanvasTool('pan');
$('showMasks').onchange = draw;
$('maskSettings').onclick = () => $('maskName').focus();
$('goCanvas').onclick = () => { canvas.focus(); canvas.scrollIntoView({block: 'center'}); };

function eventPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
          Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
}
canvas.addEventListener('pointerdown', event => {
  if (!photo || event.button !== 0 || event.isPrimary === false) return;
  if (canvasTool === 'pan') {
    panStart = {pointerId: event.pointerId, x: event.clientX, y: event.clientY, left: $('stage').scrollLeft, top: $('stage').scrollTop};
    canvas.style.cursor = 'grabbing';
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
    return;
  }
  const point = eventPoint(event), rect = canvas.getBoundingClientRect();
  const index = points.findIndex(([x, y]) => Math.hypot((x - point[0]) * rect.width, (y - point[1]) * rect.height) <= 12);
  if (index >= 0) {
    draggedPoint = index;
    dragPointer = event.pointerId;
    canvas.setPointerCapture(event.pointerId);
  } else {
    if (points.length >= 128) { feedback('A polygon supports at most 128 points.', true); return; }
    points.push(point);
  }
  canvas.focus({preventScroll: true});
  event.preventDefault();
  listMasks();
});
canvas.addEventListener('pointermove', event => {
  if (panStart) {
    if (event.pointerId !== panStart.pointerId) return;
    $('stage').scrollLeft = panStart.left + panStart.x - event.clientX;
    $('stage').scrollTop = panStart.top + panStart.y - event.clientY;
    return;
  }
  if (draggedPoint === null || event.pointerId !== dragPointer) return;
  points[draggedPoint] = eventPoint(event);
  draw();
});
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture'])
  canvas.addEventListener(type, event => {
    if (event.pointerId === dragPointer) { draggedPoint = null; dragPointer = null; }
    if (event.pointerId === panStart?.pointerId) panStart = null;
    canvas.style.cursor = panStart ? 'grabbing' : canvasTool === 'pan' ? 'grab' : 'crosshair';
  });

function saveMask() {
  const fromCanvas = ['canvas', 'canvasSave'].includes(document.activeElement?.id);
  const name = $('maskName').value.trim();
  if (!name || name.length > 80) { feedback('Give the object a name of 1–80 characters.', true); $('maskName').focus(); return; }
  if (masks.some((mask, i) => i !== editingIndex && mask.name.trim() === name)) {
    feedback('Object names must be unique.', true); $('maskName').focus(); return;
  }
  const problem = polygonError(points);
  if (problem) { feedback(problem, true); return; }
  if (editingIndex === null && masks.length >= 32) { feedback('A recipe supports at most 32 masks.', true); return; }
  rememberMasks();
  // Retain any future mask metadata when editing a known mask.
  const mask = {...(editingIndex === null ? {} : masks[editingIndex]), name,
                action: $('action').value, polygon: structuredClone(points)};
  const wasEditing = editingIndex !== null;
  if (wasEditing) masks[editingIndex] = mask;
  else masks.push(mask);
  resetDraft();
  listMasks();
  feedback(name + (wasEditing ? ' updated.' : ' added.') + ' Render to see the final composite.');
  (fromCanvas && photo ? canvas : $('maskName')).focus({preventScroll: true});
}
function undoPoint() {
  const fromCanvas = document.activeElement?.id === 'canvasUndo';
  points.pop(); draggedPoint = null; dragPointer = null; listMasks();
  if (fromCanvas && !points.length && photo) canvas.focus({preventScroll: true});
}
function cancelDraft() {
  const fromCanvas = ['canvas', 'canvasCancel'].includes(document.activeElement?.id);
  resetDraft(); listMasks(); feedback('Polygon changes cancelled. Saved masks are unchanged.');
  (fromCanvas && photo ? canvas : $('maskName')).focus({preventScroll: true});
}
$('add').onclick = saveMask;
$('undo').onclick = undoPoint;
$('cancel').onclick = cancelDraft;
$('canvasSave').onclick = saveMask;
$('canvasUndo').onclick = undoPoint;
$('canvasCancel').onclick = cancelDraft;
$('action').onchange = draw;
$('undoMask').onclick = () => changeHistory();
$('redoMask').onclick = () => changeHistory(true);

document.addEventListener('keydown', event => {
  // Native text editing (including browser undo) always belongs to the field.
  if (event.isComposing || event.target.closest('input, textarea, select, [contenteditable], [role="textbox"]')) return;
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && !event.altKey && key === 'z') {
    event.preventDefault();
    changeHistory(event.shiftKey);
  } else if (!event.ctrlKey && !event.metaKey && !event.altKey) {
    if (key === 'escape' && hasDraft()) { event.preventDefault(); cancelDraft(); }
    else if (key === 'backspace' && points.length) { event.preventDefault(); undoPoint(); }
    else if (key === 'enter' && !event.target.closest('button, a, summary, [role="button"]') && !$('add').disabled) {
      event.preventDefault(); saveMask();
    }
  }
});

function syncOpacity() {
  $('opacity').disabled = !$('override').checked;
  $('opacityValue').textContent = Math.round(Number($('opacity').value) * 100) + '%';
  updateSaveState();
}
$('override').onchange = syncOpacity;
$('opacity').oninput = syncOpacity;

function checkedConfig() {
  if (hasDraft()) { $('maskName').focus(); throw Error('Save or cancel the polygon in progress first.'); }
  const value = config();
  if ($('margin').value === '' || !Number.isFinite(value.background_margin_px) || value.background_margin_px < 0 || value.background_margin_px > 512)
    invalidField('margin', 'Context margin must be between 0 and 512 pixels.');
  clearFieldError('margin');
  return value;
}
function clearFieldError(id) {
  const message = $(id + 'Error').textContent;
  $(id).removeAttribute('aria-invalid');
  $(id).removeAttribute('aria-describedby');
  $(id + 'Error').hidden = true;
  $(id + 'Error').textContent = '';
  if (message && $('status').textContent === message) status('Setting updated. Render when ready.');
  if (message && $('feedback').textContent === message) feedback('Setting updated.');
}
function invalidField(id, message) {
  $(id).setAttribute('aria-invalid', 'true');
  $(id).setAttribute('aria-describedby', id + 'Error');
  $(id + 'Error').textContent = message;
  $(id + 'Error').hidden = false;
  $(id).focus();
  throw Error(message);
}
for (const id of ['margin', 'inputs', 'destination']) {
  $(id).addEventListener('input', () => { clearFieldError(id); updateSaveState(); });
}
$('download').onclick = () => {
  try {
    const blob = new Blob([JSON.stringify(checkedConfig(), null, 2) + '\n'], {type: 'application/json'});
    const url = URL.createObjectURL(blob), anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'mask-recipe.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    savedConfig = JSON.stringify(config());
    updateSaveState();
    feedback('Recipe downloaded.');
  } catch (error) { feedback(error.message, true); }
};

function requestError(data) {
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map(issue => {
    const field = (issue.loc ?? []).filter(part => part !== 'body').join(' → ');
    return (field ? field + ': ' : '') + (issue.msg ?? 'Invalid value');
  }).join('\n');
  return data.message ?? 'The server could not complete this request.';
}

async function jsonFetch(url, options) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(url, {...options, signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw Error(requestError(data));
    return data;
  } finally { clearTimeout(timer); }
}

function showProgress(job) {
  const event = job.progress ?? {}, progress = $('jobProgress');
  const stages = {probe: 'Checking source videos', proxies: 'Preparing previews', analysis: 'Analyzing focus',
    select: 'Selecting frames', extract: 'Extracting full-resolution frames', align: 'Aligning frames',
    fuse: 'Combining focused regions', complete: 'Finishing output'};
  const label = job.status === 'queued' ? 'Queued — waiting for a render worker'
    : stages[event.stage] ?? (event.stage ? String(event.stage).replaceAll('_', ' ') : 'Rendering');
  const measured = Number.isFinite(event.completed) && Number.isFinite(event.total) && event.total > 0;
  const count = measured ? ' · ' + event.completed + ' of ' + event.total : '';
  const selection = Number.isFinite(event.selected) && Number.isFinite(event.candidates)
    ? '\nSelected ' + event.selected + ' of ' + event.candidates + ' candidate frames.' : '';
  status(label + count + (event.message ? '\n' + event.message : '') + selection);
  progress.hidden = false;
  if (measured) { progress.max = event.total; progress.value = Math.min(event.total, Math.max(0, event.completed)); }
  else progress.removeAttribute('value');
  progress.setAttribute('aria-label', label);
}

function clearResultViews() {
  $('resultViews').replaceChildren();
  $('resultViews').hidden = true;
  $('resultCaption').textContent = '';
  $('resultCaption').hidden = true;
}

function showResultViews(job) {
  const options = [
    ['stack.png', 'Full image', 'Completed stack', 'Full rendered image. Checkerboard shows transparent areas.'],
    ['subject-detail.png', 'Subject detail', 'Rendered subject detail crop', 'Close-up of the rendered subject. Checkerboard shows transparent areas.'],
    ['comparison.jpg', 'Before & after', 'Reference on the left and rendered stack on the right', 'Reference on the left · rendered stack on the right.']
  ].filter(([name]) => Object.hasOwn(job.artifacts ?? {}, name));
  if (!options.length) return;
  const buttons = [];
  const select = index => {
    const [name, , alt, caption] = options[index];
    $('result').src = '/jobs/' + encodeURIComponent(job.id) + '/artifacts/' + encodeURIComponent(name);
    $('result').alt = alt;
    $('result').hidden = false;
    $('resultCaption').textContent = caption;
    $('resultCaption').hidden = false;
    buttons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
    updateResultState();
  };
  options.forEach(([, label], index) => {
    const button = document.createElement('button');
    button.textContent = label;
    button.onclick = () => select(index);
    buttons.push(button);
    $('resultViews').append(button);
  });
  $('resultViews').hidden = false;
  select(0);
}

function finished(job) {
  activeJob = null;
  $('run').disabled = false;
  $('jobProgress').hidden = true;
  $('links').replaceChildren();
  $('artifactLinks').replaceChildren();
  $('artifactDetails').hidden = true;
  clearResultViews();
  $('result').hidden = true;
  renderedState = null;
  $('resultState').hidden = true;
  refreshJobs();
  if (job.status !== 'completed') { activeRenderState = null; status(job.status + ': ' + (job.error ?? 'Job stopped'), true); return; }
  status('Completed. Download the result and validation artifacts below.');
  renderedState = activeRenderState;
  activeRenderState = null;
  const downloads = {'stack.png': 'Download PNG', 'stack.tiff': 'Download 16-bit TIFF', 'preview.jpg': 'Download JPEG preview'};
  for (const name of Object.keys(job.artifacts ?? {})) {
    const url = '/jobs/' + encodeURIComponent(job.id) + '/artifacts/' + encodeURIComponent(name);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.textContent = Object.hasOwn(downloads, name) ? downloads[name] : name;
    if (Object.hasOwn(downloads, name)) {
      anchor.className = 'download';
      anchor.download = name;
      $('links').append(anchor);
    } else {
      $('artifactLinks').append(anchor);
      $('artifactDetails').hidden = false;
    }
    if (name === 'reference.png') {
      const button = document.createElement('button');
      button.textContent = 'Use rendered reference for editing';
      button.onclick = () => loadPhoto(url);
      $('links').append(button);
    }
  }
  showResultViews(job);
  updateResultState();
}

async function poll(id, delay = 1000) {
  if (activeJob !== id) return;
  try {
    const job = await jsonFetch('/jobs/' + encodeURIComponent(id));
    if (activeJob !== id) return;
    if (!['queued', 'running'].includes(job.status)) { finished(job); return; }
    showProgress(job);
    setTimeout(() => poll(id, Math.min(delay * 1.3, 5000)), delay);
  } catch (error) {
    if (activeJob !== id) return;
    status('Status check failed: ' + error.message + '. Retrying…', true);
    setTimeout(() => poll(id, Math.min(delay * 2, 30000)), Math.min(delay * 2, 30000));
  }
}
$('run').onclick = async () => {
  try {
    const configuration = checkedConfig();
    const inputs = videoInputs();
    const output_dir = $('destination').value.trim();
    if (!inputs.length) invalidField('inputs', 'Add at least one video path. You can choose videos in Browse workspace.');
    clearFieldError('inputs');
    if (!output_dir) invalidField('destination', 'Choose an output directory for the rendered images.');
    clearFieldError('destination');
    $('run').disabled = true;
    $('links').replaceChildren();
    $('artifactLinks').replaceChildren();
    $('artifactDetails').hidden = true;
    clearResultViews();
    $('result').hidden = true;
    $('resultState').hidden = true;
    status('Submitting…');
    const job = await jsonFetch('/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({inputs, output_dir, config: configuration})});
    savedConfig = JSON.stringify(configuration);
    updateSaveState();
    window.history.replaceState(null, '', '?job=' + encodeURIComponent(job.id));
    activeJob = job.id;
    activeRenderState = renderState(configuration, inputs);
    if (['queued', 'running'].includes(job.status)) { showProgress(job); poll(job.id); }
    else finished(job);
  } catch (error) { $('run').disabled = false; status(error.message, true); }
};

function editorLoadState() {
  return JSON.stringify({configuration: config(), points, editingIndex,
    name: $('maskName').value, action: $('action').value,
    inputs: $('inputs').value, destination: $('destination').value, photoLoadEpoch});
}

async function loadLinkedJob() {
  const id = new URLSearchParams(location.search).get('job');
  if (id === null) return;
  if (!/^[a-f0-9]{32}$/.test(id)) { status('Invalid job link: expected a 32-character job ID.', true); return; }
  const request = ++recipeLoadEpoch, initialState = editorLoadState();
  try {
    status('Loading saved job…');
    const job = await jsonFetch('/jobs/' + encodeURIComponent(id));
    if (request !== recipeLoadEpoch) return;
    if (editorLoadState() !== initialState || hasDraft()) {
      status('Saved job was not loaded because you changed the editor. Reopen the job link to load it.', true);
      return;
    }
    if (!canReplaceRecipe()) return;
    applyRecipe(job.config, 'job ' + id);
    $('inputs').value = job.inputs.join('\n');
    $('destination').value = job.output_dir + '-edited';
    clearFieldError('inputs');
    clearFieldError('destination');
    activeRenderState = renderState();
    if (['queued', 'running'].includes(job.status)) {
      activeJob = job.id;
      $('run').disabled = true;
      showProgress(job);
      poll(job.id);
      return;
    }
    finished(job);
    if (job.status !== 'completed') {
      feedback('Saved recipe and video paths loaded. Adjust them and render to retry in a new -edited directory.');
      return;
    }
    if (Object.hasOwn(job.artifacts ?? {}, 'reference.png')) {
      loadPhoto('/jobs/' + encodeURIComponent(id) + '/artifacts/reference.png');
      status('Completed job loaded for editing. Recipe and video paths retained; output uses a new -edited directory.');
    } else status('Completed job loaded. This run has no reference.png; load its matching reference image before drawing.');
  } catch (error) {
    if (request === recipeLoadEpoch) status('Could not open saved job: ' + error.message, true);
  }
}
async function refreshJobs() {
  $('refreshJobs').disabled = true;
  try {
    const data = await jsonFetch('/jobs?limit=20');
    $('recentJobs').replaceChildren();
    for (const job of data.jobs) {
      const row = document.createElement('li'), link = document.createElement('a');
      link.href = '/?job=' + encodeURIComponent(job.id);
      link.textContent = job.output_dir + ' · ' + job.status;
      const date = new Date(typeof job.created_at === 'number' ? job.created_at * 1000 : job.created_at);
      if (job.created_at != null && !Number.isNaN(date.getTime())) link.title = date.toLocaleString();
      row.append(link);
      $('recentJobs').append(row);
    }
    $('recentStatus').textContent = data.jobs.length ? 'Open a job to edit, monitor, or retry it.' : 'No saved jobs yet.';
  } catch (error) { $('recentStatus').textContent = 'Could not load recent jobs: ' + error.message; }
  finally { $('refreshJobs').disabled = false; }
}
$('refreshJobs').onclick = refreshJobs;

let workspaceDirectory = '', workspaceParent = null, workspaceRequest = 0;
async function browseWorkspace(directory = '') {
  const request = ++workspaceRequest;
  $('workspaceStatus').textContent = 'Loading files…';
  try {
    const data = await jsonFetch('/workspace/files?directory=' + encodeURIComponent(directory));
    if (request !== workspaceRequest) return;
    workspaceDirectory = data.directory;
    workspaceParent = data.parent;
    $('workspacePath').textContent = data.directory || 'Workspace root';
    $('workspaceUp').disabled = data.parent === null;
    const list = $('workspaceFiles');
    list.replaceChildren();
    for (const [kind, entries] of [['folder', data.directories], ['video', data.videos], ['recipe', data.recipes]]) {
      for (const entry of entries) {
        const row = document.createElement('li'), button = document.createElement('button');
        button.textContent = (kind === 'folder' ? 'Open ' : kind === 'video' ? 'Add video: ' : 'Load recipe: ') + entry.name;
        button.onclick = async () => {
          if (kind === 'folder') { browseWorkspace(entry.path); return; }
          if (kind === 'video') {
            const inputs = $('inputs').value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
            const relativePath = path => {
              const root = (data.workspace_root ?? '').replace(/\/$/, '');
              if (root && path.startsWith(root + '/')) path = path.slice(root.length + 1);
              return path.replace(/^(\.\/)+/, '');
            };
            const selected = inputs.some(path => relativePath(path) === relativePath(entry.path));
            if (!selected) {
              inputs.push(entry.path);
              $('inputs').value = inputs.join('\n');
            }
            clearFieldError('inputs');
            updateResultState();
            $('workspaceStatus').textContent = (selected ? 'Video already selected: ' : 'Video added: ') + entry.path;
            return;
          }
          if (hasDraft()) { feedback('Save or cancel the current polygon before loading a recipe.', true); return; }
          const recipeRequest = ++recipeLoadEpoch;
          button.disabled = true;
          try {
            const recipeFile = await jsonFetch('/workspace/recipe?path=' + encodeURIComponent(entry.path));
            if (recipeRequest === recipeLoadEpoch && canReplaceRecipe()) applyRecipe(recipeFile.config, recipeFile.name);
          } catch (error) {
            if (recipeRequest === recipeLoadEpoch) feedback('Could not load recipe: ' + error.message, true);
          }
          finally { button.disabled = false; }
        };
        row.append(button);
        list.append(row);
      }
    }
    $('workspaceStatus').textContent = data.truncated ? 'Showing a limited listing. Open a folder to narrow it down.'
      : list.children.length ? 'Choose a folder, video, or recipe.' : 'No videos, recipes, or folders here.';
  } catch (error) {
    if (request === workspaceRequest) $('workspaceStatus').textContent = 'Could not browse workspace: ' + error.message;
  }
}
$('workspaceUp').onclick = () => { if (workspaceParent !== null) browseWorkspace(workspaceParent); };
$('workspaceRefresh').onclick = () => browseWorkspace(workspaceDirectory);
listMasks();
savedConfig = JSON.stringify(config());
updateSaveState();
window.addEventListener('beforeunload', event => {
  if (hasDraft() || JSON.stringify(config()) !== savedConfig) {
    event.preventDefault();
    event.returnValue = '';
  }
});
loadLinkedJob();
refreshJobs();
browseWorkspace();
</script>
</body>
</html>'''

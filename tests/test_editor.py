import json
import re
import shutil
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from macrostitch import api


POLYGON = [[.1, .1], [.9, .1], [.5, .9]]


def test_editor_endpoint(tmp_path):
    with TestClient(api.create_app(tmp_path)) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/html')
        assert 'Reference canvas' in response.text
        assert 'getBoundingClientRect' in response.text
        assert 'innerHTML' not in response.text
        assert 'id="run"' in response.text
        assert "recipe.subject_polygon" in response.text
        assert 'reference.png' in response.text


@pytest.mark.parametrize('values', [
    {'background_opacity': -.1}, {'background_opacity': 1.1},
    {'background_opacity': float('nan')}, {'background_margin_px': -1},
    {'background_margin_px': 513}, {'object_masks': []},
    {'object_masks': [{'name': 'a', 'action': 'invalid', 'polygon': POLYGON}]},
    {'object_masks': [{'name': '', 'action': 'stack', 'polygon': POLYGON}]},
    {'object_masks': [{'name': 'a', 'action': 'stack', 'polygon': POLYGON}] * 2},
    {'object_masks': [{'name': 'a', 'action': 'stack', 'polygon': [[0, 0], [1, 1]]}]},
    {'object_masks': [{'name': 'a', 'action': 'exclude', 'polygon': [[0, 0], [1, 1], [2, 0]]}]},
])
def test_mask_schema_rejects_invalid(values):
    with pytest.raises(ValueError):
        api.StackOptions(**values)


def test_mask_schema_accepts_controls_without_legacy_polygon():
    options = api.StackOptions(background_mode='transparent', background_opacity=.3,
                               background_margin_px=8,
                               object_masks=[{'name': 'flower', 'action': 'stack', 'polygon': POLYGON}])
    assert options.model_dump()['background_opacity'] == .3
    api.Config(**options.model_dump()).validate()


def test_cli_mask_controls_override_recipe(tmp_path, monkeypatch):
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'background_opacity': .9, 'background_margin_px': 2,
                                  'threads': 2, 'proxy_width': 512}))
    masks = tmp_path / 'masks.json'
    rows = [{'name': 'leaf', 'action': 'reference', 'polygon': POLYGON}]
    masks.write_text(json.dumps(rows))
    calls = []
    monkeypatch.setattr(api, 'run_stack', lambda *args, **kwargs: calls.append(args) or {})
    monkeypatch.setattr(sys, 'argv', ['macrostitch', 'stack', 'clip.mov', '--output', 'masked',
                                    '--config', str(recipe), '--background-opacity', '0',
                                    '--background-margin-px', '9', '--object-masks', str(masks)])
    api.main()
    config = calls[0][2]
    assert config.background_opacity == 0
    assert config.background_margin_px == 9
    assert list(config.object_masks) == rows
    assert config.threads == 2 and config.proxy_width == 512


def test_mask_job_validation_and_propagation(tmp_path, monkeypatch):
    import time
    (tmp_path / 'clip.mov').write_bytes(b'video')
    calls = []
    monkeypatch.setattr(api, 'run_stack', lambda inputs, output, config, progress:
                        calls.append(config) or {'artifacts': {}})
    with TestClient(api.create_app(tmp_path)) as client:
        base = {'inputs': ['clip.mov'], 'output_dir': 'masked'}
        bad = client.post('/jobs', json={**base, 'config': {'object_masks': [
            {'name': 'object', 'action': 'stack', 'polygon': [[0, 0], [1, 1], [2, 0]]}]}})
        assert bad.status_code == 422
        config = {'background_opacity': .25, 'background_margin_px': 4, 'object_masks': [
            {'name': 'object', 'action': 'exclude', 'polygon': POLYGON}]}
        response = client.post('/jobs', json={**base, 'config': config})
        assert response.status_code == 202
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = client.get('/jobs/' + response.json()['id']).json()
            if job['status'] not in ('queued', 'running'):
                break
            time.sleep(.01)
        assert job['status'] == 'completed'
        assert calls[0].background_opacity == .25
        assert calls[0].background_margin_px == 4
        assert calls[0].object_masks[0]['action'] == 'exclude'



# Execute the real inline controller, with a small DOM substitute. Assertions below
# exercise saved recipe behavior rather than requiring specific source formatting.
# Real browser layout, drawing, and pointer capture are covered by visual QA.
def run_editor_js(checks):
    from macrostitch.editor import EDITOR_HTML
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for editor controller behavior checks')
    script = re.search(r'<script>(.*?)</script>', EDITOR_HTML, re.S).group(1)
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
class Element {
  constructor(id = '') {
    this.id = id; this.value = ''; this.children = []; this.style = {};
    this.hidden = false; this.disabled = false; this.checked = false;
    this.attributes = {}; this.listeners = {}; this.textContent = '';
    this.classList = {toggle() {}};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  getContext() { return new Proxy({}, {get: () => () => {}}); }
  getBoundingClientRect() { return this.rect ?? {left: 0, top: 0, width: 1000, height: 500}; }
  setPointerCapture() {}
  focus() { document.activeElement = this; }
  closest() { return null; }
}
const elements = new Map();
const document = {
  listeners: {},
  confirm: () => true,
  history: {replaceState(state, title, url) { this.url = url; }},
  getElementById(id) { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); },
  createElement() { return new Element(); },
  addEventListener(name, callback) { this.listeners[name] = callback; }
};
let responses = {};
const context = vm.createContext({assert, document, window: document, responses, structuredClone, URLSearchParams, URL, Blob,
  AbortController, console, location: {search: ''}, Image: class {},
  setTimeout: () => 1, clearTimeout() {},
  fetch: async url => ({ok: true, json: async () => structuredClone(responses[url] ??
    (url.startsWith('/workspace/files') ? {directory: '', parent: null, directories: [], videos: [], recipes: []}
    : {jobs: [], total: 0}))})
});
document.getElementById('action').value = 'stack';
document.getElementById('margin').value = '0';
document.getElementById('opacity').value = '1';
document.getElementById('showMasks').checked = true;
vm.runInContext(SCRIPT, context);
(async () => {
  // Let startup fetches complete before each isolated behavior scenario.
  await new Promise(resolve => setImmediate(resolve));
  await vm.runInContext('(async () => {' + CHECKS + '})()', context);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    program = 'const SCRIPT = ' + json.dumps(script) + ';\nconst CHECKS = ' + json.dumps(checks) + ';\n' + harness
    result = subprocess.run([node, '-e', program], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_editor_edits_masks_without_losing_recipe_and_supports_history():
    run_editor_js(r"""
      const original = {threads: 3, custom_future_setting: {preserve: true},
        subject_polygon: [[0, 0], [1, 0], [.5, 1]], background_opacity: .4,
        object_masks: [{name: 'leaf', action: 'stack', polygon: [[.1, .1], [.8, .1], [.5, .9]], extra: 'keep'}]};
      applyRecipe(original, 'fixture');
      editMask(0);
      $('maskName').value = 'foreground';
      $('action').value = 'reference';
      points[0] = [.2, .3];
      assert.throws(checkedConfig, /Save or cancel/);
      saveMask();
      const updated = checkedConfig();
      assert.equal(updated.object_masks.length, 1);
      assert.equal(updated.object_masks[0].name, 'foreground');
      assert.equal(updated.object_masks[0].action, 'reference');
      assert.equal(updated.object_masks[0].extra, 'keep');
      assert.equal(updated.object_masks[0].polygon[0][0], .2);
      assert.equal(updated.threads, 3);
      assert.equal(updated.custom_future_setting.preserve, true);
      assert.equal(updated.background_opacity, .4);
      assert.equal(JSON.stringify(updated.subject_polygon), JSON.stringify(original.subject_polygon));
      assert.equal(original.object_masks[0].name, 'leaf');
      changeHistory();
      assert.equal(config().object_masks[0].name, 'leaf');
      changeHistory(true);
      assert.equal(config().object_masks[0].name, 'foreground');
      editMask(0);
      points[0] = [.7, .7];
      cancelDraft();
      assert.equal(config().object_masks[0].polygon[0][0], .2);
      removeMask(0);
      assert.equal(config().object_masks, undefined);
      changeHistory();
      assert.equal(config().object_masks[0].name, 'foreground');
      applyRecipe({}, 'replacement');
      assert.equal(undoMasks.length, 0);
      assert.equal(redoMasks.length, 0);
    """)


def test_editor_rejects_bad_masks_and_duplicate_names_without_changing_saved_data():
    run_editor_js(r"""
      const polygon = [[0, 0], [1, 0], [.5, 1]];
      applyRecipe({object_masks: [{name: 'one', action: 'stack', polygon},
        {name: 'two', action: 'exclude', polygon}]}, 'fixture');
      editMask(0);
      $('maskName').value = ' two ';
      saveMask();
      assert.equal(masks[0].name, 'one');
      assert.match($('feedback').textContent, /unique/);
      cancelDraft();
      for (const bad of [{name: 'x', action: 'toString', polygon},
                         {name: 'x', action: 'stack', polygon: [[0,0], [1,1]]}]) {
        assert.throws(() => applyRecipe({object_masks: [bad]}, 'bad'), /Invalid object masks/);
        assert.equal(masks.length, 2);
      }
      applyRecipe({object_masks: Array.from({length: 32}, (_, i) => ({name: String(i), action: 'stack', polygon}))}, 'full');
      editMask(0);
      assert.equal($('add').disabled, false);
      $('maskName').value = 'renamed at capacity';
      saveMask();
      assert.equal(masks.length, 32);
      assert.equal(masks[0].name, 'renamed at capacity');
    """)


def test_editor_pointer_editing_keeps_normalized_coordinates_when_zoomed():
    run_editor_js(r"""
      applyRecipe({object_masks: [{name: 'one', action: 'stack', polygon: [[.1, .1], [.8, .1], [.5, .9]]}]}, 'fixture');
      photo = {};
      canvas.width = 2000; canvas.height = 1000;
      editMask(0);
      setZoom(4);
      canvas.rect = {left: -100, top: -200, width: 4000, height: 2000};
      const event = {clientX: 300, clientY: 0, pointerId: 1, button: 0, preventDefault() {}};
      canvas.listeners.pointerdown(event);
      canvas.listeners.pointermove({...event, clientX: 900, clientY: 800});
      canvas.listeners.pointerup(event);
      assert.equal(points.length, 3);
      assert.equal(points[0][0], .25);
      assert.equal(points[0][1], .5);
      saveMask();
      assert.equal(config().object_masks[0].polygon[0][0], .25);
      assert.equal(canvas.style.width, '4000px');
      assert.equal(eventPoint({clientX: -1000, clientY: 9000})[0], 0);
      assert.equal(eventPoint({clientX: -1000, clientY: 9000})[1], 1);
    """)


def test_editor_keyboard_shortcuts_leave_text_editing_to_browser():
    run_editor_js(r"""
      masks = [{name: 'one', action: 'stack', polygon: [[0,0], [1,0], [.5,1]]}];
      removeMask(0);
      let prevented = false;
      const event = {key: 'z', ctrlKey: true, shiftKey: false, altKey: false,
        target: {closest: () => ({})}, preventDefault() { prevented = true; }};
      document.listeners.keydown(event);
      assert.equal(prevented, false);
      assert.equal(masks.length, 0);
      event.target = {closest: () => null};
      document.listeners.keydown(event);
      assert.equal(prevented, true);
      assert.equal(masks.length, 1);
      document.listeners.keydown({...event, shiftKey: true});
      assert.equal(masks.length, 0);
      points = [[.1,.1]];
      document.listeners.keydown({...event, ctrlKey: false, key: 'Escape'});
      assert.equal(points.length, 0);
    """)


def test_editor_progress_is_readable_and_stage_based():
    run_editor_js(r"""
      showProgress({status: 'running', progress: {stage: 'align', completed: 3, total: 10, message: 'Matching frames'}});
      assert.equal($('status').textContent, 'Aligning frames · 3 of 10\nMatching frames');
      assert.equal($('jobProgress').max, 10);
      assert.equal($('jobProgress').value, 3);
      assert.equal($('jobProgress').hidden, false);
      showProgress({status: 'queued'});
      assert.match($('status').textContent, /waiting for a render worker/);
      showProgress({status: 'running', progress: {stage: 'select', selected: 7, candidates: 90}});
      assert.match($('status').textContent, /Selected 7 of 90 candidate frames/);
      finished({status: 'failed', error: 'No usable frames'});
      assert.equal($('jobProgress').hidden, true);
      assert.match($('status').textContent, /No usable frames/);
    """)


def test_editor_polygon_errors_are_local_and_do_not_mutate_saved_masks():
    run_editor_js(r"""
      const invalid = [
        [[0,0], [.5,.5], [1,1]],
        [[0,0], [1,0], [.5,1], [0,0]],
        [[0,0], [1,.7], [0,1], [.7,0]]
      ];
      $('maskName').value = 'bad';
      for (const poly of invalid) {
        points = poly;
        saveMask();
        assert.equal(masks.length, 0);
        assert.ok($('feedback').textContent.length > 10);
      }
      assert.equal(polygonError([[0,0], [1,0], [1,1], [.5,.5], [0,1]]), null);
      assert.match(requestError({detail: [{loc: ['body', 'config', 'threads'], msg: 'Must be positive'}]}),
        /config → threads: Must be positive/);
      assert.equal(requestError({detail: 'Output already exists'}), 'Output already exists');
    """)


def test_editor_reopens_completed_running_and_failed_jobs():
    run_editor_js(r"""
      const id = 'a'.repeat(32), url = '/jobs/' + id;
      location.search = '?job=' + id;
      responses[url] = {id, status: 'completed', config: {threads: 5}, inputs: ['video.mov'],
        output_dir: 'outputs/original', artifacts: {'reference.png': 'ignored filesystem path'}};
      await loadLinkedJob();
      assert.equal(config().threads, 5);
      assert.equal($('inputs').value, 'video.mov');
      assert.equal($('destination').value, 'outputs/original-edited');
      assert.match($('status').textContent, /loaded for editing/);
      responses[url].status = 'running';
      responses[url].progress = {stage: 'align', completed: 2, total: 5};
      await loadLinkedJob();
      assert.equal(activeJob, id);
      assert.equal($('run').disabled, true);
      assert.match($('status').textContent, /Aligning frames/);
      responses[url].status = 'failed';
      responses[url].error = 'Fixture failure';
      await loadLinkedJob();
      assert.equal(activeJob, null);
      assert.equal($('run').disabled, false);
      assert.match($('status').textContent, /Fixture failure/);
      assert.match($('feedback').textContent, /render to retry/);
      location.search = '?job=../invalid';
      await loadLinkedJob();
      assert.match($('status').textContent, /Invalid job link/);
    """)


def test_editor_workspace_picker_adds_videos_once_and_loads_recipes():
    run_editor_js(r"""
      responses['/workspace/files?directory=clips'] = {directory: 'clips', parent: '', directories: [],
        videos: [{name: 'one.mov', path: 'clips/one.mov'}], recipes: [{name: 'settings.json', path: 'clips/settings.json'}]};
      responses['/workspace/recipe?path=clips%2Fsettings.json'] = {name: 'settings.json', config: {threads: 6}};
      await browseWorkspace('clips');
      assert.equal($('workspacePath').textContent, 'clips');
      assert.equal($('workspaceUp').disabled, false);
      const video = $('workspaceFiles').children[0].children[0];
      await video.onclick();
      await video.onclick();
      assert.equal($('inputs').value, 'clips/one.mov');
      const recipeButton = $('workspaceFiles').children[1].children[0];
      await recipeButton.onclick();
      assert.equal(config().threads, 6);
      assert.equal($('inputs').value, 'clips/one.mov');
      points = [[.2,.2]];
      responses['/workspace/recipe?path=clips%2Fsettings.json'].config.threads = 9;
      await recipeButton.onclick();
      assert.equal(config().threads, 6);
      assert.match($('feedback').textContent, /Save or cancel/);
      responses['/jobs?limit=20'] = {jobs: [{id: 'b'.repeat(32), status: 'completed', output_dir: 'outputs/one', created_at: 1792627200}]};
      await refreshJobs();
      const link = $('recentJobs').children[0].children[0];
      assert.equal(link.href, '/?job=' + 'b'.repeat(32));
      assert.match(link.textContent, /outputs\/one · completed/);
      assert.equal(link.title, new Date(1792627200 * 1000).toLocaleString());
    """)


def test_editor_warns_before_leaving_with_unsaved_recipe_or_draft():
    run_editor_js(r"""
      let prevented = false;
      const event = {preventDefault() { prevented = true; }};
      document.listeners.beforeunload(event);
      assert.equal(prevented, false);
      points = [[.1,.1]];
      document.listeners.beforeunload(event);
      assert.equal(prevented, true);
      prevented = false;
      resetDraft();
      $('margin').value = '12';
      document.listeners.beforeunload(event);
      assert.equal(prevented, true);
      prevented = false;
      applyRecipe({background_margin_px: 12}, 'saved');
      document.listeners.beforeunload(event);
      assert.equal(prevented, false);
    """)



def test_editor_fit_contains_whole_reference_and_zoom_tracks_resized_stage():
    run_editor_js(r"""
      photo = {};
      canvas.width = 4000; canvas.height = 2250;
      $('stage').rect = {left: 0, top: 0, width: 810, height: 290};
      setZoom(1);
      const fittedWidth = parseFloat(canvas.style.width);
      assert.ok(fittedWidth <= 810);
      assert.ok(Math.abs(fittedWidth * canvas.height / canvas.width - 290) < 1e-8);
      setZoom(2);
      assert.equal(parseFloat(canvas.style.width), fittedWidth * 2);
      setZoom(4);
      assert.equal(parseFloat(canvas.style.width), fittedWidth * 4);
      $('stage').rect = {left: 0, top: 0, width: 500, height: 600};
      fitCanvas();
      assert.equal(parseFloat(canvas.style.width), 2000);
      setZoom(1);
      assert.equal(parseFloat(canvas.style.width), 500);
      assert.equal($('stage').scrollLeft, 0);
      assert.equal($('stage').scrollTop, 0);
      canvas.width = 1000; canvas.height = 2000;
      fitCanvas();
      assert.equal(parseFloat(canvas.style.width), 300);
    """)


def test_editor_workspace_video_dedupes_absolute_and_dot_relative_paths():
    run_editor_js(r"""
      responses['/workspace/files?directory=clips'] = {directory: 'clips', parent: '',
        workspace_root: '/my/workspace', directories: [], recipes: [],
        videos: [{name: 'one.mov', path: 'clips/one.mov'}]};
      await browseWorkspace('clips');
      const video = $('workspaceFiles').children[0].children[0];
      for (const existing of ['/my/workspace/clips/one.mov', './clips/one.mov', '././clips/one.mov', '  /my/workspace/clips/one.mov\n\n']) {
        $('inputs').value = existing;
        await video.onclick();
        assert.equal($('inputs').value, existing);
      }
      $('inputs').value = '/my/workspace-other/clips/one.mov';
      await video.onclick();
      assert.equal($('inputs').value, '/my/workspace-other/clips/one.mov\nclips/one.mov');
    """)



def test_editor_recipe_load_can_be_cancelled_without_losing_changes():
    run_editor_js(r"""
      applyRecipe({threads: 2}, 'initial');
      $('margin').value = '11';
      window.confirm = () => false;
      const upload = {files: [{name: 'new.json', text: async () => '{"threads":7}'}], value: 'new.json'};
      await $('recipe').onchange({target: upload});
      assert.equal(config().threads, 2);
      assert.equal(config().background_margin_px, 11);
      assert.equal(upload.value, '');
      responses['/workspace/files?directory='] = {directory: '', parent: null, directories: [], videos: [],
        recipes: [{name: 'new.json', path: 'new.json'}]};
      responses['/workspace/recipe?path=new.json'] = {name: 'new.json', config: {threads: 9}};
      await browseWorkspace();
      await $('workspaceFiles').children[0].children[0].onclick();
      assert.equal(config().threads, 2);
      assert.equal(config().background_margin_px, 11);
      window.confirm = () => true;
      await $('workspaceFiles').children[0].children[0].onclick();
      assert.equal(config().threads, 9);
      assert.equal(config().background_margin_px, 0);
    """)


def test_editor_submitted_job_updates_recovery_url():
    run_editor_js(r"""
      const id = 'c'.repeat(32);
      responses['/jobs'] = {id, status: 'completed', artifacts: {}};
      $('inputs').value = 'clip.mov';
      $('destination').value = 'outputs/test';
      await $('run').onclick();
      assert.equal(window.history.url, '?job=' + id);
      assert.equal($('run').disabled, false);
      assert.match($('status').textContent, /Completed/);
    """)


def test_editor_latest_workspace_recipe_wins_when_requests_finish_out_of_order():
    run_editor_js(r"""
      responses['/workspace/files?directory='] = {directory: '', parent: null, directories: [], videos: [],
        recipes: [{name: 'A.json', path: 'A.json'}, {name: 'B.json', path: 'B.json'}]};
      await browseWorkspace();
      const a = $('workspaceFiles').children[0].children[0], b = $('workspaceFiles').children[1].children[0];
      let resolveA, resolveB;
      fetch = url => new Promise(resolve => { if (url.includes('A.json')) resolveA = resolve; else resolveB = resolve; });
      const first = a.onclick(), second = b.onclick();
      resolveB({ok: true, json: async () => ({name: 'B.json', config: {threads: 7}})});
      await second;
      resolveA({ok: true, json: async () => ({name: 'A.json', config: {threads: 2}})});
      await first;
      assert.equal(config().threads, 7);
      assert.match($('recipeInfo').textContent, /B.json/);
      assert.equal(a.disabled, false);
      assert.equal(b.disabled, false);
    """)


def test_editor_file_and_workspace_recipe_loads_share_latest_intent():
    run_editor_js(r"""
      responses['/workspace/files?directory='] = {directory: '', parent: null, directories: [], videos: [],
        recipes: [{name: 'workspace.json', path: 'workspace.json'}]};
      await browseWorkspace();
      const button = $('workspaceFiles').children[0].children[0];
      let resolveFile, resolveWorkspace;
      const uploaded = {files: [{name: 'upload.json', text: () => new Promise(resolve => { resolveFile = resolve; })}], value: 'upload.json'};
      fetch = () => new Promise(resolve => { resolveWorkspace = resolve; });
      const olderFile = $('recipe').onchange({target: uploaded}), newerWorkspace = button.onclick();
      resolveWorkspace({ok: true, json: async () => ({name: 'workspace.json', config: {threads: 5}})});
      await newerWorkspace;
      resolveFile('{"threads":2}'); await olderFile;
      assert.equal(config().threads, 5);
      assert.equal(uploaded.value, '');
      const olderWorkspace = button.onclick(), newerFile = $('recipe').onchange({target: uploaded});
      resolveFile('{"threads":8}'); await newerFile;
      resolveWorkspace({ok: true, json: async () => ({name: 'workspace.json', config: {threads: 3}})});
      await olderWorkspace;
      assert.equal(config().threads, 8);
    """)


def test_editor_parallel_uploads_and_stale_failures_preserve_latest_selection():
    run_editor_js(r"""
      let finishOld, finishNew;
      const input = {value: 'old.json', files: [{name: 'old.json', text: () => new Promise(resolve => { finishOld = resolve; })}]};
      const oldLoad = $('recipe').onchange({target: input});
      input.value = 'new.json';
      input.files = [{name: 'new.json', text: () => new Promise(resolve => { finishNew = resolve; })}];
      const newLoad = $('recipe').onchange({target: input});
      finishOld('{'); await oldLoad;
      assert.equal(input.value, 'new.json');
      finishNew('{"threads":9}'); await newLoad;
      assert.equal(config().threads, 9);
      assert.equal(input.value, '');
      assert.equal($('feedback').textContent, 'Recipe loaded. Existing mask history has been reset.');
    """)


@pytest.mark.parametrize('edit', [
    "points = [[.1,.1],[.8,.1],[.5,.8]];",
    "$('margin').value = '12';",
    "$('inputs').value = 'new-video.mov';",
    "$('destination').value = 'outputs/new-work';",
    "loadPhoto('new-reference.png');",
])
def test_editor_delayed_linked_job_cannot_erase_intervening_edits(edit):
    run_editor_js(r"""
      const id = 'a'.repeat(32); location.search = '?job=' + id;
      let finishJob;
      fetch = url => url === '/jobs/' + id ? new Promise(resolve => { finishJob = resolve; })
        : Promise.resolve({ok: true, json: async () => ({jobs: [], total: 0})});
      const reopening = loadLinkedJob();
    """ + edit + r"""
      const edited = editorLoadState();
      finishJob({ok: true, json: async () => ({id, status: 'completed', config: {threads: 2},
        inputs: ['old.mov'], output_dir: 'outputs/old', artifacts: {}})});
      await reopening;
      assert.equal(editorLoadState(), edited);
      assert.match($('status').textContent, /not loaded because you changed/);
      assert.equal(activeJob, null);
    """)


def test_editor_linked_job_response_cannot_replace_a_new_recipe_load():
    run_editor_js(r"""
      const id = 'a'.repeat(32); location.search = '?job=' + id;
      let finishJob;
      fetch = () => new Promise(resolve => { finishJob = resolve; });
      const reopening = loadLinkedJob();
      await $('recipe').onchange({target: {files: [{name: 'latest.json', text: async () => '{"threads":7}'}], value: 'latest.json'}});
      finishJob({ok: true, json: async () => ({id, status: 'completed', config: {threads: 2},
        inputs: ['old.mov'], output_dir: 'outputs/old', artifacts: {}})});
      await reopening;
      assert.equal(config().threads, 7);
      assert.match($('recipeInfo').textContent, /latest.json/);
      assert.equal($('inputs').value, '');
      assert.match($('status').textContent, /Recipe loaded/);
    """)


def test_editor_latest_reference_image_wins_and_stale_urls_are_released():
    run_editor_js(r"""
      const images = [], revoked = [];
      Image = class { constructor() { this.naturalWidth = 2000; this.naturalHeight = 1000; images.push(this); } };
      URL.revokeObjectURL = url => revoked.push(url);
      loadPhoto('blob:old', true); loadPhoto('blob:new', true);
      images[1].onload();
      assert.equal(photo, images[1]);
      images[0].onload();
      assert.equal(photo, images[1]);
      assert.equal(revoked.join(','), 'blob:new,blob:old');
      loadPhoto('blob:stale-error', true); loadPhoto('blob:current', true);
      images[3].onload();
      const message = $('feedback').textContent;
      images[2].onerror();
      assert.equal(photo, images[3]);
      assert.equal($('feedback').textContent, message);
      assert.equal(revoked.at(-1), 'blob:stale-error');
    """)

"""Viewport navigation behavior for the mask editor."""

from test_editor import run_editor_js


ZOOM_LAYOUT = r"""
  photo = {};
  canvas.width = 2000; canvas.height = 1000;
  const stage = $('stage');
  stage.rect = {left: 20, top: 40, width: 800, height: 500};
  stage.clientWidth = 800; stage.clientHeight = 500;
  stage.scrollLeft = 0; stage.scrollTop = 0;
  canvas.getBoundingClientRect = () => {
    const width = parseFloat(canvas.style.width) || 800;
    const height = width * canvas.height / canvas.width;
    return {left: 20 + Math.max(0, (800 - width) / 2) - stage.scrollLeft,
      top: 40 - stage.scrollTop, width, height};
  };
"""


def test_editor_zoom_keeps_the_inspected_image_center():
    run_editor_js(ZOOM_LAYOUT + r"""
      setZoom(1);
      setZoom(2);
      assert.equal(stage.scrollLeft, 400);
      assert.equal(stage.scrollTop, 150);
      stage.scrollLeft = 650; stage.scrollTop = 250;
      setZoom(4);
      assert.equal(stage.scrollLeft, 1700);
      assert.equal(stage.scrollTop, 750);
      setZoom(2);
      assert.equal(stage.scrollLeft, 650);
      assert.equal(stage.scrollTop, 250);
      setZoom(1);
      assert.equal(stage.scrollLeft, 0);
      assert.equal(stage.scrollTop, 0);
    """)


def test_editor_zoom_centers_portrait_canvas_and_clamps_fitted_dimensions():
    run_editor_js(ZOOM_LAYOUT + r"""
      canvas.width = 1000; canvas.height = 2000;
      setZoom(1);
      assert.equal(parseFloat(canvas.style.width), 250);
      setZoom(2);
      assert.equal(stage.scrollLeft, 0);
      assert.equal(stage.scrollTop, 250);
      setZoom(4);
      assert.equal(stage.scrollLeft, 100);
      assert.equal(stage.scrollTop, 750);
      stage.scrollLeft = 200; stage.scrollTop = 1500;
      setZoom(2);
      assert.equal(stage.scrollLeft, 0);
      assert.equal(stage.scrollTop, 500);
    """)


def test_editor_pan_does_not_change_polygon_and_can_return_to_drawing():
    run_editor_js(r"""
      photo = {}; canvas.width = 2000; canvas.height = 1000;
      points = [[.1,.1], [.8,.1], [.5,.8]];
      const original = JSON.stringify(points), stage = $('stage');
      stage.scrollLeft = 300; stage.scrollTop = 200;
      const event = {clientX: 500, clientY: 300, pointerId: 1, button: 0, preventDefault() {}};
      $('panTool').onclick();
      canvas.listeners.pointerdown(event);
      canvas.listeners.pointermove({...event, clientX: 400, clientY: 250});
      assert.equal(stage.scrollLeft, 400); assert.equal(stage.scrollTop, 250);
      assert.equal(JSON.stringify(points), original);
      canvas.listeners.pointercancel(event);
      canvas.listeners.pointermove({...event, clientX: 100, clientY: 100});
      assert.equal(stage.scrollLeft, 400);
      assert.equal(canvas.style.cursor, 'grab');
      $('drawTool').onclick();
      canvas.listeners.pointerdown({...event, clientX: 300, clientY: 300});
      assert.equal(points.length, 4);
      assert.equal($('canvasPointCount').textContent, '4 points');
      $('canvasUndo').onclick();
      assert.equal(points.length, 3);
      $('maskName').value = 'test';
      $('canvasSave').onclick();
      assert.equal(masks.length, 1); assert.equal(points.length, 0);
      assert.match($('canvasFeedback').textContent, /test added/);
    """)


def test_editor_render_validation_focuses_fields_and_is_visible_beside_render():
    run_editor_js(r"""
      let submissions = 0;
      fetch = async () => { submissions++; throw Error('unexpected submission'); };
      $('inputs').value = ''; $('destination').value = 'outputs/test';
      await $('run').onclick();
      assert.equal(document.activeElement, $('inputs'));
      assert.equal($('inputs').attributes['aria-invalid'], 'true');
      assert.match($('renderFeedback').textContent, /at least one video/);
      $('inputs').value = 'clip.mov'; $('inputs').listeners.input();
      assert.equal($('inputs').attributes['aria-invalid'], undefined);
      $('destination').value = '  ';
      await $('run').onclick();
      assert.equal(document.activeElement, $('destination'));
      assert.match($('renderFeedback').textContent, /output directory/);
      $('destination').value = 'outputs/test'; $('margin').value = '';
      await $('run').onclick();
      assert.equal(document.activeElement, $('margin'));
      assert.match($('renderFeedback').textContent, /0 and 512/);
      assert.equal(submissions, 0);
      assert.equal($('run').disabled, false);
    """)


def test_editor_enter_on_disclosure_does_not_save_polygon():
    run_editor_js(r"""
      points = [[.1,.1], [.8,.1], [.5,.8]];
      $('maskName').value = 'draft'; draw();
      let prevented = false;
      document.listeners.keydown({key: 'Enter', target: {
        closest: selector => selector.includes('summary') ? {} : null
      }, preventDefault() { prevented = true; }});
      assert.equal(prevented, false); assert.equal(masks.length, 0);
      assert.equal(points.length, 3);
    """)


def test_editor_saved_state_tracks_changes_history_and_drafts():
    run_editor_js(r"""
      applyRecipe({}, 'fixture');
      assert.equal($('saveState').textContent, 'No unsaved recipe changes');
      $('margin').value = '12'; $('margin').listeners.input();
      assert.match($('saveState').textContent, /Unsaved recipe changes/);
      $('margin').value = '0'; $('margin').listeners.input();
      assert.equal($('saveState').textContent, 'No unsaved recipe changes');
      points = [[.1,.1], [.8,.1], [.5,.8]]; draw();
      assert.match($('saveState').textContent, /Polygon in progress/);
      $('maskName').value = 'one'; saveMask();
      assert.match($('saveState').textContent, /Unsaved recipe changes/);
      changeHistory();
      assert.equal($('saveState').textContent, 'No unsaved recipe changes');
    """)


def test_editor_mask_removal_keeps_keyboard_focus_in_the_mask_workflow():
    run_editor_js(r"""
      const polygon = [[.1,.1], [.8,.1], [.5,.8]];
      applyRecipe({object_masks: [{name: 'one', action: 'stack', polygon},
        {name: 'two', action: 'stack', polygon}]}, 'fixture');
      removeMask(0);
      assert.equal(document.activeElement.attributes['aria-label'], 'Edit two');
      removeMask(0);
      assert.equal(document.activeElement, $('maskName'));
      $('maskSettings').onclick();
      assert.equal(document.activeElement, $('maskName'));
    """)


def test_editor_hidden_saved_overlays_keep_active_polygon_visible():
    run_editor_js(r"""
      photo = {}; canvas.width = 2000; canvas.height = 1000;
      const polygon = [[.1,.1], [.8,.1], [.5,.8]];
      applyRecipe({subject_polygon: polygon, object_masks: [{name: 'one', action: 'stack', polygon}]}, 'fixture');
      points = [[.2,.2], [.7,.2], [.5,.7]];
      const drawn = []; drawPoly = poly => drawn.push(poly);
      $('showMasks').checked = false; $('showMasks').onchange();
      assert.equal(drawn.length, 1); assert.equal(drawn[0], points);
      drawn.length = 0; $('showMasks').checked = true; $('showMasks').onchange();
      assert.equal(drawn.length, 3);
      assert.equal(config().object_masks.length, 1);
    """)


def test_editor_programmatic_recovery_clears_invalid_fields():
    run_editor_js(r"""
      $('inputs').value = ''; $('destination').value = 'outputs/test';
      await $('run').onclick();
      responses['/workspace/files?directory='] = {directory: '', parent: null, directories: [], recipes: [],
        videos: [{name: 'clip.mov', path: 'clip.mov'}]};
      await browseWorkspace(); await $('workspaceFiles').children[0].children[0].onclick();
      assert.equal($('inputs').value, 'clip.mov');
      assert.equal($('inputs').attributes['aria-invalid'], undefined);
      $('margin').value = ''; assert.throws(checkedConfig);
      applyRecipe({background_margin_px: 12}, 'valid');
      assert.equal($('margin').attributes['aria-invalid'], undefined);
      $('destination').setAttribute('aria-invalid', 'true');
      const id = 'd'.repeat(32); location.search = '?job=' + id;
      responses['/jobs/' + id] = {id, status: 'completed', config: {}, inputs: ['clip.mov'], output_dir: 'outputs/original', artifacts: {}};
      await loadLinkedJob();
      assert.equal($('destination').attributes['aria-invalid'], undefined);
    """)


def test_editor_edit_switches_from_pan_and_other_pointers_cannot_move_points():
    run_editor_js(r"""
      photo = {}; canvas.width = 2000; canvas.height = 1000;
      applyRecipe({object_masks: [{name: 'one', action: 'stack', polygon: [[.1,.1], [.8,.1], [.5,.8]]}]}, 'fixture');
      setCanvasTool('pan'); editMask(0);
      assert.equal(canvasTool, 'draw');
      const event = {clientX: 100, clientY: 50, pointerId: 1, button: 0, preventDefault() {}};
      canvas.listeners.pointerdown(event);
      const original = JSON.stringify(points);
      canvas.listeners.pointermove({...event, pointerId: 2, clientX: 500});
      assert.equal(JSON.stringify(points), original);
      canvas.listeners.pointerup({...event, pointerId: 2});
      assert.equal(draggedPoint, 0);
      canvas.listeners.pointermove({...event, clientX: 200});
      assert.equal(points[0][0], .2);
      canvas.listeners.lostpointercapture(event);
      canvas.listeners.pointermove({...event, clientX: 300});
      assert.equal(points[0][0], .2);
      $('stage').scrollLeft = 100; $('stage').scrollTop = 100;
      setCanvasTool('pan');
      for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        canvas.listeners.pointerdown(event);
        canvas.listeners.pointermove({...event, pointerId: 2, clientX: 500});
        assert.equal($('stage').scrollLeft, 100);
        canvas.listeners[type](event);
        canvas.listeners.pointermove({...event, clientX: 500});
        assert.equal($('stage').scrollLeft, 100);
      }
    """)


def test_editor_result_marks_recipe_or_video_changes_and_groups_downloads():
    run_editor_js(r"""
      $('inputs').value = 'clip.mov';
      activeRenderState = renderState();
      finished({id: 'a'.repeat(32), status: 'completed', artifacts: {
        'stack.png': '/unused', 'stack.tiff': '/unused', 'preview.jpg': '/unused',
        'reference.png': '/unused', 'selection.json': '/unused'
      }});
      assert.equal($('resultState').hidden, false);
      assert.match($('resultState').textContent, /matches/);
      assert.equal($('links').children[0].textContent, 'Download PNG');
      assert.equal($('links').children[0].download, 'stack.png');
      assert.equal($('artifactLinks').children.length, 2);
      assert.equal($('artifactDetails').hidden, false);
      $('margin').value = '12'; $('margin').listeners.input();
      assert.match($('resultState').textContent, /last render/);
      savedConfig = JSON.stringify(config()); updateSaveState();
      assert.equal($('saveState').textContent, 'No unsaved recipe changes');
      assert.match($('resultState').textContent, /last render/);
      $('margin').value = '0'; $('margin').listeners.input();
      assert.match($('resultState').textContent, /matches/);
      $('inputs').value = 'other.mov'; $('inputs').listeners.input();
      assert.match($('resultState').textContent, /last render/);
      finished({status: 'failed', error: 'fixture'});
      assert.equal($('resultState').hidden, true);
      assert.equal($('result').hidden, true);
      assert.equal($('artifactDetails').hidden, true);
    """)


def test_editor_edits_during_submission_are_not_claimed_saved_or_rendered():
    run_editor_js(r"""
      let resolveJob;
      fetch = url => url === '/jobs' ? new Promise(resolve => { resolveJob = resolve; })
        : Promise.resolve({ok: true, json: async () => ({jobs: [], total: 0})});
      $('inputs').value = 'clip.mov'; $('destination').value = 'outputs/test';
      const submission = $('run').onclick();
      $('margin').value = '12'; $('margin').listeners.input();
      resolveJob({ok: true, json: async () => ({id: 'f'.repeat(32), status: 'completed', artifacts: {'stack.png': '/unused'}})});
      await submission;
      assert.equal(config().background_margin_px, 12);
      assert.match($('saveState').textContent, /Unsaved recipe changes/);
      assert.match($('resultState').textContent, /last render/);
      assert.equal($('run').disabled, false);
    """)


def test_editor_canvas_save_cancel_and_last_undo_keep_keyboard_focus():
    run_editor_js(r"""
      photo = {}; canvas.width = 2000; canvas.height = 1000;
      points = [[.1,.1], [.8,.1], [.5,.8]]; $('maskName').value = 'one'; draw();
      $('canvasSave').focus(); saveMask();
      assert.equal(document.activeElement, canvas);
      points = [[.2,.2]]; draw(); $('canvasCancel').focus(); cancelDraft();
      assert.equal(document.activeElement, canvas);
      points = [[.2,.2]]; draw(); $('canvasUndo').focus(); undoPoint();
      assert.equal(document.activeElement, canvas);
      points = [[.1,.1], [.8,.1], [.5,.8]]; $('maskName').value = 'two'; draw();
      $('add').focus(); saveMask();
      assert.equal(document.activeElement, $('maskName'));
      $('margin').value = ''; assert.throws(checkedConfig);
      assert.equal($('marginError').hidden, false);
      assert.equal($('margin').attributes['aria-describedby'], 'marginError');
      $('margin').value = '12'; $('margin').listeners.input();
      assert.equal($('marginError').hidden, true);
      assert.equal($('margin').attributes['aria-describedby'], undefined);
    """)


def test_editor_error_recovery_and_history_buttons_do_not_leave_stale_feedback_or_focus():
    run_editor_js(r"""
      $('inputs').value = ''; $('destination').value = 'outputs/test';
      await $('run').onclick();
      assert.match($('status').textContent, /at least one video/);
      $('inputs').value = 'clip.mov'; $('inputs').listeners.input();
      assert.equal($('status').textContent, 'Setting updated. Render when ready.');
      const polygon = [[.1,.1], [.8,.1], [.5,.8]];
      applyRecipe({object_masks: [{name: 'one', action: 'stack', polygon}]}, 'fixture');
      removeMask(0);
      $('undoMask').focus(); changeHistory();
      assert.equal(document.activeElement, $('redoMask'));
      assert.equal(masks.length, 1);
      changeHistory(true);
      assert.equal(document.activeElement, $('undoMask'));
      assert.equal(masks.length, 0);
    """)

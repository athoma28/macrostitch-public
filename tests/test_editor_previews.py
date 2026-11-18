"""Completed render preview selection and lifecycle."""

from test_editor import run_editor_js


def test_editor_result_views_switch_image_and_preserve_stale_notice():
    run_editor_js(r"""
      $('inputs').value = 'clip.mov';
      activeRenderState = renderState();
      const id = 'a'.repeat(32);
      finished({id, status: 'completed', artifacts: {
        'comparison.jpg': '', 'subject-detail.png': '', 'stack.png': ''
      }});
      const views = $('resultViews').children;
      assert.equal(views.map(button => button.textContent).join('|'), 'Full image|Subject detail|Before & after');
      assert.ok($('result').src.endsWith('/stack.png'));
      assert.equal(views[0].attributes['aria-pressed'], 'true');
      assert.equal(views[1].attributes['aria-pressed'], 'false');
      fetch = () => { throw Error('Preview switches must not fetch controller data'); };
      views[1].onclick();
      assert.ok($('result').src.endsWith('/subject-detail.png'));
      assert.equal($('result').alt, 'Rendered subject detail crop');
      assert.equal(views[0].attributes['aria-pressed'], 'false');
      assert.equal(views[1].attributes['aria-pressed'], 'true');
      assert.match($('resultState').textContent, /matches/);
      $('margin').value = '12'; $('margin').listeners.input();
      views[2].onclick();
      assert.ok($('result').src.endsWith('/comparison.jpg'));
      assert.match($('resultCaption').textContent, /Reference on the left.*stack on the right/);
      assert.equal(views[2].attributes['aria-pressed'], 'true');
      assert.match($('resultState').textContent, /last render/);
      assert.equal(config().background_margin_px, 12);
    """)


def test_editor_result_views_fall_back_and_clear_on_failure_or_submission():
    run_editor_js(r"""
      const id = 'b'.repeat(32);
      activeRenderState = renderState();
      finished({id, status: 'completed', artifacts: {'comparison.jpg': ''}});
      assert.equal($('resultViews').children.length, 1);
      assert.ok($('result').src.endsWith('/comparison.jpg'));
      assert.equal($('result').hidden, false);
      finished({id, status: 'failed', error: 'fixture'});
      assert.equal($('resultViews').children.length, 0);
      assert.equal($('resultViews').hidden, true);
      assert.equal($('resultCaption').hidden, true);
      assert.equal($('result').hidden, true);
      finished({id, status: 'completed', artifacts: {'selection.json': ''}});
      assert.equal($('resultViews').hidden, true);
      assert.equal($('result').hidden, true);
      finished({id, status: 'completed', artifacts: {'subject-detail.png': ''}});
      assert.equal($('resultViews').children[0].textContent, 'Subject detail');
      $('inputs').value = 'clip.mov'; $('destination').value = 'outputs/test';
      fetch = async () => { throw Error('Submission failed'); };
      await $('run').onclick();
      assert.equal($('resultViews').children.length, 0);
      assert.equal($('resultCaption').textContent, '');
      assert.equal($('resultCaption').hidden, true);
      assert.equal($('result').hidden, true);
    """)

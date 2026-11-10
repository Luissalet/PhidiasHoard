// Phidias's Hoard studio: library of game models, projects, the 3-D view and the sculpting controls.
import * as THREE from 'three';
import { OrbitControls } from '/vendor/three/OrbitControls.js';
import { $, el, api, toast, fail, modal, confirmBox, nf, store } from '/ui.js';

const call = (name, args = {}) => api('/api/ui/call', { method: 'POST', body: { name, arguments: args } });
const state = { project: null, which: 'before', library: null, projects: [], poll: null, loaded: '' };
const RECIPES = [['auto', 'Automática'], ['normal', 'Mapa de normales'], ['relief', 'Relieve'], ['lines', 'Solo líneas'], ['steps', 'Escalones'], ['flat', 'Liso'], ['drop', 'Quitar']];
const WHY = { before: 'Original', sculpt: 'Esculpido', closed: 'Sólido cerrado' };

// ------------------------------------------------------------------ 3-D view
const view = (() => {
  const host = $('viewport');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100000);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xfff4e0, 0x2a2622, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  scene.add(key);
  const material = new THREE.MeshStandardMaterial({ color: 0xd9c8ad, roughness: 0.82, metalness: 0.0, side: THREE.DoubleSide });
  let mesh = null;
  let dirty = true;
  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    dirty = true;
  }
  new ResizeObserver(resize).observe(host);
  controls.addEventListener('change', () => { dirty = true; });
  (function loop() {
    requestAnimationFrame(loop);
    if (controls.update() || dirty) {
      key.position.copy(camera.position).add(new THREE.Vector3(0, 0, camera.position.length() * 0.6));
      renderer.render(scene, camera);
      dirty = false;
    }
  })();
  function show(buffer, keepCamera) {
    const head = new Uint32Array(buffer, 0, 2);
    const points = new Float32Array(buffer, 8, head[0] * 3);
    const index = new Uint32Array(buffer, 8 + head[0] * 12, head[1] * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(points, 3));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
    mesh = new THREE.Mesh(geo, material);
    scene.add(mesh);
    if (!keepCamera) {
      const s = geo.boundingSphere;
      const d = s.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
      controls.target.copy(s.center);
      camera.position.copy(s.center).add(new THREE.Vector3(0.62, -0.62, 0.48).normalize().multiplyScalar(d));
      camera.near = s.radius / 500; camera.far = s.radius * 50;
      camera.updateProjectionMatrix();
    }
    dirty = true;
    return { points: head[0], triangles: head[1] };
  }
  function clear() { if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh = null; dirty = true; } }
  return { show, clear, resize };
})();

async function loadMesh(keepCamera = false) {
  const p = state.project;
  if (!p) return;
  const key = `${p.id}/${state.which}/${p.updated}`;
  $('loading').hidden = false;
  try {
    const res = await fetch(`/api/projects/${p.id}/mesh/${state.which}.bin?t=${Date.now()}`);
    if (!res.ok) throw new Error('La vista 3D de este resultado no está disponible (malla demasiado grande): descarga el STL.');
    const info = view.show(await res.arrayBuffer(), keepCamera && state.loaded.startsWith(p.id + '/'));
    state.loaded = key;
    $('view-info').textContent = `${WHY[state.which]} · ${nf.format(info.triangles)} triángulos en la vista`;
  } catch (err) { fail(err); } finally { $('loading').hidden = true; }
}

// ------------------------------------------------------------------ library and projects
function human(bytes) { return bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} kB`; }

async function loadLibrary(refresh = false) {
  const list = $('lib-list');
  list.replaceChildren(el('div', { class: 'hint', text: 'Leyendo las carpetas…' }));
  try {
    state.library = await call('library_scan', { refresh });
  } catch (err) { fail(err); return; }
  drawLibrary();
}

function drawLibrary() {
  const lib = state.library, list = $('lib-list');
  if (!lib) return;
  const q = $('lib-search').value.trim().toLowerCase();
  list.replaceChildren();
  if (!lib.dirs.length) {
    list.append(el('div', { class: 'hint' }, 'Todavía no hay ninguna carpeta de modelos. ',
      el('button', { class: 'btn small', onclick: openSettings }, 'Elegir carpetas')));
    return;
  }
  for (const m of lib.missing) list.append(el('div', { class: 'hint', text: `No encuentro la carpeta ${m}` }));
  let group = null, shown = 0;
  for (const e of lib.entries) {
    if (q && !(`${e.name} ${e.group}`.toLowerCase().includes(q))) continue;
    if (e.group !== group) { group = e.group; list.append(el('div', { class: 'group', text: group })); }
    shown += 1;
    list.append(el('button', { class: `item${state.project && e.project_id === state.project.id ? ' active' : ''}`, title: e.path + (e.inner ? ` → ${e.inner}` : ''),
      onclick: () => openEntry(e) },
      el('span', { class: 'name', text: e.name }), el('span', { class: 'meta', text: human(e.bytes) }),
      el('span', { class: `tag${e.project_id ? ' done' : ''}`, text: e.format })));
  }
  if (!shown) list.append(el('div', { class: 'hint', text: q ? 'Nada con ese nombre.' : 'No hay modelos OBJ, DAE, GLB o glTF en esas carpetas.' }));
}

async function openEntry(e) {
  try {
    if (e.project_id) return openProject(e.project_id);
    toast(`Leyendo ${e.name}…`);
    const p = await call('project_create', { source: e.path, inner: e.inner || '' });
    e.project_id = p.id;
    await loadProjects();
    setProject(p);
    drawLibrary();
  } catch (err) { fail(err); }
}

async function loadProjects() {
  try { state.projects = (await call('project_list')).projects; } catch (err) { fail(err); return; }
  const list = $('project-list');
  list.replaceChildren();
  if (!state.projects.length) list.append(el('div', { class: 'hint', text: 'Abre un escenario de la biblioteca para empezar.' }));
  for (const p of state.projects) {
    const done = p.results.closed ? 'cerrado' : (p.results.sculpt ? 'esculpido' : 'sin esculpir');
    list.append(el('div', { class: 'row' },
      el('button', { class: `item${state.project && p.id === state.project.id ? ' active' : ''}`, onclick: () => openProject(p.id) },
        el('img', { class: 'thumb', src: `/api/projects/${p.id}/picture/${p.results.sculpt ? 'sculpt' : 'before'}.png`, alt: '', loading: 'lazy' }),
        el('span', { class: 'name', text: p.name }), el('span', { class: `tag${p.results.sculpt ? ' done' : ''}`, text: done })),
      el('button', { class: 'btn small danger', title: 'Borrar el proyecto (el modelo de origen no se toca)', onclick: () => removeProject(p) }, '×')));
  }
}

async function removeProject(p) {
  if (!await confirmBox('Borrar el proyecto', `Se borran «${p.name}» y sus resultados. El modelo de origen no se toca.`, 'Borrar', true)) return;
  try {
    await call('project_delete', { project_id: p.id });
    if (state.project && state.project.id === p.id) setProject(null);
    await loadProjects(); await loadLibrary();
  } catch (err) { fail(err); }
}

async function openProject(id) {
  try { setProject(await call('project_get', { project_id: id })); } catch (err) { fail(err); }
}

// ------------------------------------------------------------------ the open project
function setProject(p, keepView = false) {
  const changed = !state.project || !p || state.project.id !== p.id;
  state.project = p;
  store('project', p ? p.id : '');
  $('right').hidden = !p;
  $('empty').hidden = !!p;
  $('project-title').textContent = p ? p.name : '';
  if (!p) { view.clear(); $('view-info').textContent = ''; $('resultbar').replaceChildren(); refreshButtons(); return; }
  const r = p.results || {};
  if (changed && !keepView) state.which = r.closed ? 'closed' : (r.sculpt ? 'sculpt' : 'before');
  if (!r[state.which] && state.which !== 'before') state.which = r.sculpt ? 'sculpt' : 'before';
  const s = p.settings;
  $('s-size').value = s.size_mm; $('s-edge').value = s.min_edge_mm; $('s-ground').checked = !!s.ground; $('s-base').value = s.base_mm;
  $('s-plinth').value = s.plinth || 'none';
  const faces = $('s-faces');
  if (![...faces.options].some((o) => +o.value === +s.max_faces)) faces.append(el('option', { value: s.max_faces, text: `${nf.format(s.max_faces)}` }));
  faces.value = s.max_faces;
  drawMaterials();
  drawResult();
  refreshButtons();
  loadMesh(!changed);
  document.querySelectorAll('#project-list .item, #lib-list .item').forEach((n) => n.classList.remove('active'));
  if (p.job) watchJob(p.job.id);
  view.resize();
}

function refreshButtons() {
  const p = state.project, r = p ? (p.results || {}) : {}, busy = !!state.poll;
  for (const b of $('view-seg').children) {
    const w = b.dataset.which;
    b.disabled = !p || (w !== 'before' && !r[w]);
    b.classList.toggle('active', w === state.which);
  }
  $('btn-sculpt').disabled = !p || busy;
  $('btn-sculpt').textContent = r.sculpt ? 'Volver a esculpir' : 'Esculpir';
  $('btn-close').disabled = !p || busy || !r.sculpt;
  $('btn-download').disabled = !p || !r.sculpt;
  $('btn-export').disabled = !p || !r.sculpt;
}

function drawResult() {
  const p = state.project, bar = $('resultbar');
  bar.replaceChildren();
  if (!p) return;
  const src = p.source;
  bar.append(el('span', {}, 'Origen: ', el('b', { text: `${nf.format(src.triangles)} triángulos` }), ` · ${src.format.toUpperCase()}`));
  const r = (p.results || {})[state.which === 'before' ? 'sculpt' : state.which];
  if (!r) { bar.append(el('span', { text: 'Sin esculpir todavía.' })); return; }
  bar.append(el('span', {}, `${WHY[state.which === 'before' ? 'sculpt' : state.which]}: `, el('b', { text: `${nf.format(r.triangles_total)} triángulos` })));
  bar.append(el('span', {}, 'Medidas: ', el('b', { text: `${r.size_mm.map((v) => v.toFixed(1)).join(' × ')} mm` })));
  if (r.edge_mm) bar.append(el('span', { class: r.edge_mm > r.edge_asked_mm * 1.05 ? 'warn' : '' },
    `Detalle: ${r.edge_mm} mm${r.edge_mm > r.edge_asked_mm * 1.05 ? ` (pedías ${r.edge_asked_mm}; sube el presupuesto de triángulos)` : ''}`));
  bar.append(el('span', { class: r.open_edges ? 'warn' : 'ok', text: r.open_edges ? `${nf.format(r.open_edges)} bordes abiertos (ciérralo con «Cerrar malla» si tu laminador se queja)` : 'Malla cerrada' }));
  bar.append(el('span', { text: `${human(r.file_bytes)} · ${r.seconds} s` }));
}

function drawMaterials() {
  const p = state.project, list = $('mat-list');
  const mats = Object.entries(p.materials).sort((a, b) => b[1].share - a[1].share);
  $('mat-count').textContent = `${mats.length}`;
  list.replaceChildren();
  for (const [name, m] of mats) {
    const recipe = el('select', { class: 'input', 'aria-label': `Receta de ${name}`, onchange: (ev) => saveMaterial(name, { recipe: ev.target.value }) },
      RECIPES.map(([v, t]) => el('option', { value: v, text: t, selected: v === m.recipe })));
    const depth = el('input', { class: 'input', type: 'number', min: '0', max: '20', step: '0.1', value: m.depth_mm, title: 'Profundidad del relieve (mm)',
      'aria-label': `Profundidad de ${name}`, onchange: (ev) => saveMaterial(name, { depth_mm: +ev.target.value }) });
    const cut = m.transparent > 0.01 && m.transparent < 0.99 ? el('label', { class: 'check', title: 'Quitar lo que la textura deja transparente (vallas, hojas, barandillas) y dar grosor al resto' },
      el('input', { type: 'checkbox', checked: !!m.cutout, onchange: (ev) => saveMaterial(name, { cutout: ev.target.checked }) }), el('span', { text: 'Recortar por transparencia' })) : null;
    const tex = m.texture[0] ? `${m.texture[0]}×${m.texture[1]}` : 'sin textura';
    list.append(el('div', { class: `mat${m.recipe === 'drop' ? ' dropped' : ''}` },
      el('button', { class: 'pic', title: 'Ver textura, normales y mapa de alturas', onclick: () => openMaterial(name) },
        el('img', { class: 'thumb', src: `/api/projects/${p.id}/materials/thumb?name=${encodeURIComponent(name)}`, alt: '', loading: 'lazy' })),
      el('div', {},
        el('div', { class: 'mname', text: name, title: name }),
        el('div', { class: 'mmeta', text: `${(m.share * 100).toFixed(1)} % · ${tex}${m.normal_map ? ' · normales' : ''}${m.why ? ` · ${m.why}` : ''}` }),
        el('div', { class: 'mctl' }, recipe, depth), cut)));
  }
}

async function saveMaterial(name, change) {
  try {
    let materials = { [name]: change };
    if (name === '*') {                                   // everything except what is being removed (shadows, flames)
      materials = {};
      for (const [n, m] of Object.entries(state.project.materials)) if (m.recipe !== 'drop') materials[n] = change;
    }
    await call('material_set', { project_id: state.project.id, materials });
    const p = await call('project_get', { project_id: state.project.id });
    state.project = p;
    drawMaterials();
  } catch (err) { fail(err); }
}

async function openMaterial(name) {
  const p = state.project, m = p.materials[name];
  let recipe = m.recipe;
  const img = el('img', { class: 'sheet', alt: 'Textura, mapa de normales y mapa de alturas' });
  const src = () => { img.src = `/api/projects/${p.id}/materials/sheet?name=${encodeURIComponent(name)}&recipe=${recipe}&t=${Date.now()}`; };
  const bar = el('div', { class: 'recipes' }, RECIPES.filter(([v]) => v !== 'drop').map(([v, t]) =>
    el('button', { class: 'btn small', type: 'button', onclick: () => { recipe = v; src(); } }, t)));
  src();
  const r = await modal({ title: name, wide: true, body: [bar, img,
    el('div', { class: 'legend' }, el('span', { text: 'Textura' }), el('span', { text: 'Mapa de normales' }), el('span', { text: 'Alturas (claro = sale, oscuro = se talla)' }))],
    buttons: [{ label: 'Cerrar', value: null }, { label: 'Usar la receta que se ve', value: 'use', kind: 'primary' }] });
  if (r) saveMaterial(name, { recipe });
}

let saveTimer = null;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const p = state.project;
    if (!p) return;
    try {
      const before = p.settings.size_mm;
      const np = await call('project_set', { project_id: p.id, size_mm: +$('s-size').value, min_edge_mm: +$('s-edge').value, max_faces: +$('s-faces').value,
        ground: $('s-ground').checked, base_mm: +$('s-base').value, plinth: $('s-plinth').value });
      state.project = np;
      if (np.settings.size_mm !== before && state.which === 'before') loadMesh(true);
    } catch (err) { fail(err); }
  }, 350);
}

// ------------------------------------------------------------------ jobs
function watchJob(id) {
  clearInterval(state.poll);
  $('job').hidden = false;
  const tick = async () => {
    let j;
    try { j = await call('job_status', { job_id: id }); } catch (err) { clearInterval(state.poll); state.poll = null; $('job').hidden = true; refreshButtons(); return; }
    $('job-stage').textContent = `${j.kind === 'close' ? 'Cerrando' : 'Esculpiendo'} · ${j.stage}`;
    $('job-fill').style.width = `${Math.round(j.progress * 100)}%`;
    if (j.state === 'done' || j.state === 'error') {
      clearInterval(state.poll); state.poll = null; $('job').hidden = true;
      if (j.state === 'error') toast(j.error, 'err', 12000);
      else toast(j.kind === 'close' ? 'Sólido cerrado listo.' : 'Esculpido listo.', 'ok');
      if (state.project && state.project.id === j.project_id) {
        const p = await call('project_get', { project_id: j.project_id });
        if (j.state === 'done') state.which = j.kind === 'close' ? 'closed' : 'sculpt';
        setProject(p, true);
      }
      loadProjects();
    }
  };
  state.poll = setInterval(tick, 1000);
  refreshButtons();
  tick();
}

async function startJob(tool) {
  clearTimeout(saveTimer);
  try {
    const j = await call(tool, { project_id: state.project.id });
    watchJob(j.id);
  } catch (err) { fail(err); }
}

// ------------------------------------------------------------------ settings
async function openSettings() {
  let s, st;
  try { [s, st] = await Promise.all([call('settings_get'), call('status')]); } catch (err) { fail(err); return; }
  const dirs = el('textarea', { class: 'input', rows: '4', placeholder: 'Una carpeta por línea' });
  dirs.value = s.library_dirs.join('\n');
  const voxel = el('input', { class: 'input', type: 'number', min: '0.05', max: '5', step: '0.05', value: s.voxel_mm });
  const shell = el('input', { class: 'input', type: 'number', min: '0', max: '20', step: '0.1', value: s.shell_mm });
  const blender = el('input', { class: 'input', type: 'text', value: s.blender, placeholder: st.blender || 'No encuentro Blender: escribe la ruta de blender.exe' });
  const r = await modal({ title: 'Ajustes', body: [
    el('label', {}, el('span', { text: 'Carpetas con modelos de juegos (OBJ, DAE, GLB, glTF o zips)' }), dirs),
    el('label', { class: 'field' }, el('span', { text: 'Cerrar malla: tamaño de vóxel (mm)' }), voxel),
    el('label', { class: 'field' }, el('span', { text: 'Cerrar malla: grosor de láminas sueltas (mm)' }), shell),
    el('label', {}, el('span', { text: `Blender${st.blender ? ' (encontrado solo)' : ''}` }), blender)],
    buttons: [{ label: 'Cancelar', value: null }, { label: 'Guardar', value: 'ok', kind: 'primary' }] });
  if (!r) return;
  try {
    await call('settings_set', { changes: { library_dirs: dirs.value.split('\n').map((x) => x.trim()).filter(Boolean), voxel_mm: +voxel.value, shell_mm: +shell.value, blender: blender.value.trim() } });
    loadLibrary(true);
  } catch (err) { fail(err); }
}

// ------------------------------------------------------------------ wiring
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === t));
  $('pane-library').hidden = t.dataset.tab !== 'library';
  $('pane-projects').hidden = t.dataset.tab !== 'projects';
}));
$('lib-search').addEventListener('input', drawLibrary);
$('lib-refresh').addEventListener('click', () => loadLibrary(true));
$('btn-settings').addEventListener('click', openSettings);
for (const id of ['s-size', 's-edge', 's-faces', 's-ground', 's-base', 's-plinth']) $(id).addEventListener('change', saveSettings);
for (const b of $('view-seg').children) b.addEventListener('click', () => { state.which = b.dataset.which; refreshButtons(); drawResult(); loadMesh(true); });
$('btn-sculpt').addEventListener('click', () => startJob('sculpt_start'));
$('btn-close').addEventListener('click', () => startJob('close_start'));
$('btn-download').addEventListener('click', () => {
  const which = state.which === 'closed' ? 'closed' : 'sculpt';
  window.location.href = `/api/projects/${state.project.id}/download/${which}.stl`;
});
$('btn-export').addEventListener('click', async () => {
  try {
    const r = await call('result_export', { project_id: state.project.id, result: state.which === 'closed' ? 'closed' : 'sculpt' });
    toast(`Guardado en ${r.path}`, 'ok', 9000);
  } catch (err) { fail(err); }
});
$('bulk-recipe').addEventListener('change', async (ev) => {
  if (!ev.target.value || !state.project) return;
  await saveMaterial('*', { recipe: ev.target.value });
  ev.target.value = '';
});
$('bulk-depth').addEventListener('change', async (ev) => {
  if (ev.target.value === '' || !state.project) return;
  await saveMaterial('*', { depth_mm: +ev.target.value });
  ev.target.value = '';
});

(async function boot() {
  await Promise.all([loadLibrary(), loadProjects()]);
  const last = store('project');
  if (last && state.projects.some((p) => p.id === last)) openProject(last);
})();

/* ═══════════════════════════════════════════════════════════════
   NHL Predictions — front-end app
   ═══════════════════════════════════════════════════════════════ */
const root = document.getElementById('pageRoot');
const CY = '#00B4D8', GREEN = '#10B981', YELLOW = '#F59E0B', RED = '#EF4444', BLUE = '#3B82F6';
let charts = [];

// ── utils ────────────────────────────────────────────────
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct0 = x => (x == null ? '—' : `${Math.round(x * 100)}%`);
const signed = (x, d = 1) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${Number(x).toFixed(d)}`);
async function api(path) { const r = await fetch(path); if (!r.ok) throw new Error(`${path} → ${r.status}`); return r.json(); }
function destroyCharts() { charts.forEach(c => c.destroy()); charts = []; }
function spinner() { return '<div class="spinner"></div>'; }
function notice(msg, cls = 'info') { return `<div class="notice ${cls}">${esc(msg)}</div>`; }
function confClass(c) { return c >= 65 ? 'conf-high' : c >= 55 ? 'conf-med' : 'conf-low'; }
function chipClass(form) { return 'chip ' + form.toLowerCase().replace(/\s+/g, '-'); }

const CHART_DEFAULTS = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 900, easing: 'easeOutQuart' },
  plugins: { legend: { labels: { color: '#F4F7FB', font: { family: 'Barlow' } } } },
  scales: {
    x: { ticks: { color: '#93A1B5' }, grid: { color: 'rgba(255,255,255,0.06)' } },
    y: { ticks: { color: '#93A1B5' }, grid: { color: 'rgba(255,255,255,0.06)' } },
  },
};
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

if (window.Chart) {
  Chart.defaults.font.family = 'Barlow';
  Chart.defaults.color = '#93A1B5';
  Chart.defaults.elements.bar.borderRadius = 6;
  Chart.defaults.elements.bar.borderSkipped = false;
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(13,21,38,0.95)';
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.12)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
}

// transient toast (bottom-right)
let _toastEl = null, _toastTimer = null;
function toast(msg) {
  if (!_toastEl) { _toastEl = el('div', 'toast'); document.body.appendChild(_toastEl); }
  _toastEl.textContent = msg;
  requestAnimationFrame(() => _toastEl.classList.add('show'));
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => _toastEl.classList.remove('show'), 2600);
}

// ── NHL team colours (vivid-on-dark) → [primary, secondary] ──
const TEAM_COLORS = {
  'Anaheim Ducks': ['#F47A38', '#B9975B'], 'Boston Bruins': ['#FCB514', '#111111'],
  'Buffalo Sabres': ['#2F6FE0', '#FCB514'], 'Calgary Flames': ['#F0453E', '#F1BE48'],
  'Carolina Hurricanes': ['#E1213A', '#7B99AC'], 'Chicago Blackhawks': ['#E51A38', '#FF671B'],
  'Colorado Avalanche': ['#A23A57', '#4C90C4'], 'Columbus Blue Jackets': ['#2E5FA3', '#E1213A'],
  'Dallas Stars': ['#1AA85A', '#8F8F8C'], 'Detroit Red Wings': ['#E23140', '#D8DBE0'],
  'Edmonton Oilers': ['#FF580A', '#2F6FE0'], 'Florida Panthers': ['#E03A3E', '#B9975B'],
  'Los Angeles Kings': ['#B0B7BC', '#E2E4E6'], 'Minnesota Wild': ['#1EA362', '#C8324A'],
  'Montreal Canadiens': ['#E11B38', '#2F6FE0'], 'Montréal Canadiens': ['#E11B38', '#2F6FE0'],
  'Nashville Predators': ['#FFB81C', '#2F6FE0'], 'New Jersey Devils': ['#E23140', '#111111'],
  'New York Islanders': ['#F26722', '#2F6FE0'], 'New York Rangers': ['#2A6FE0', '#E1213A'],
  'Ottawa Senators': ['#E1213A', '#C69214'], 'Philadelphia Flyers': ['#FA4616', '#111111'],
  'Pittsburgh Penguins': ['#FCB514', '#D8DBE0'], 'San Jose Sharks': ['#10C4C4', '#E57200'],
  'Seattle Kraken': ['#6BD3E0', '#99D9D9'], 'St. Louis Blues': ['#3A78E0', '#FCB514'],
  'St Louis Blues': ['#3A78E0', '#FCB514'], 'Tampa Bay Lightning': ['#3A78E0', '#D8DBE0'],
  'Toronto Maple Leafs': ['#2C7FE0', '#D8DBE0'], 'Vancouver Canucks': ['#2F7FE0', '#10A85A'],
  'Vegas Golden Knights': ['#D4AF57', '#8C9196'], 'Washington Capitals': ['#E1213A', '#2F6FE0'],
  'Winnipeg Jets': ['#2A6FE0', '#8C9196'], 'Utah Hockey Club': ['#6CACE4', '#8C9196'],
  'Utah Mammoth': ['#6CACE4', '#8C9196'], 'Arizona Coyotes': ['#8C2633', '#DDCBA4'],
};
const teamPair = n => TEAM_COLORS[n] || ['#22D3EE', '#3B82F6'];
const teamCol = n => teamPair(n)[0];
const teamDot = n => `<span class="team-dot" style="color:${teamCol(n)}"></span>${esc(n)}`;

// ── count-up (animates the first number inside an element) ──
function animateCounts(scope) {
  scope.querySelectorAll('[data-count]:not([data-done])').forEach(el => {
    const raw = el.textContent;
    const m = raw.match(/-?\d[\d,]*\.?\d*/);
    if (!m) return;
    el.setAttribute('data-done', '1');
    if (REDUCED) return;
    const numStr = m[0].replace(/,/g, '');
    const target = parseFloat(numStr);
    const decimals = (numStr.split('.')[1] || '').length;
    const pre = raw.slice(0, m.index), post = raw.slice(m.index + m[0].length);
    const dur = 900, start = performance.now();
    const fmt = v => pre + (decimals ? v.toFixed(decimals) : Math.round(v).toLocaleString()) + post;
    el.textContent = fmt(0);
    (function tick(now) {
      const t = Math.min((now - start) / dur, 1), e = 1 - Math.pow(1 - t, 3);
      el.textContent = fmt(target * e);
      if (t < 1) requestAnimationFrame(tick); else el.textContent = raw;
    })(performance.now());
  });
}

// ── scroll-reveal (stagger) ──
const revealObs = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); revealObs.unobserve(e.target); } });
}, { threshold: 0.06 });

// ── confidence ring gauge ──
function confGauge(pct) {
  const r = 40, c = 2 * Math.PI * r;
  const col = pct >= 65 ? 'var(--green)' : pct >= 55 ? 'var(--yellow)' : 'var(--red)';
  const off = c * (1 - pct / 100);
  return `<div class="conf-gauge"><svg width="92" height="92" viewBox="0 0 92 92">
    <circle class="cg-track" cx="46" cy="46" r="${r}" fill="none" stroke-width="7"/>
    <circle class="cg-val" cx="46" cy="46" r="${r}" fill="none" stroke="${col}" stroke-width="7"
      stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${REDUCED ? off.toFixed(1) : c.toFixed(1)}" data-off="${off.toFixed(1)}"/>
    </svg><div class="cg-txt"><div class="cg-num" style="color:${col}">${pct.toFixed(0)}%</div><div class="cg-lbl">Conf</div></div></div>`;
}

// ── post-render decoration: reveal + counters + gauges + prob bars ──
function postRender() {
  const items = [...root.querySelectorAll('.hero, .kpi, .card, .chart-box, .table-wrap, .accordion details')];
  items.forEach((n, i) => {
    if (!n.classList.contains('reveal')) {
      n.classList.add('reveal');
      n.style.transitionDelay = Math.min(i * 35, 280) + 'ms';
      revealObs.observe(n);
    }
  });
  animateCounts(root);
  if (!REDUCED) {
    requestAnimationFrame(() => {
      root.querySelectorAll('.cg-val[data-off]').forEach(c => { c.style.strokeDashoffset = c.getAttribute('data-off'); });
      root.querySelectorAll('.prob-bar > span[data-w]').forEach(s => { s.style.width = s.getAttribute('data-w'); });
    });
  } else {
    root.querySelectorAll('.prob-bar > span[data-w]').forEach(s => { s.style.width = s.getAttribute('data-w'); });
  }
}

// ── sortable table ───────────────────────────────────────
// cols: [{key, label, num, render}]
function sortableTable(rows, cols, opts = {}) {
  const wrap = el('div', 'table-wrap' + (opts.tall ? ' tbl-tall' : ''));
  const table = el('table');
  const thead = el('thead');
  const trh = el('tr');
  let sortKey = opts.sortKey || null, sortDir = opts.sortDir || 'desc';
  cols.forEach(c => {
    const th = el('th', 'sortable' + (c.num ? ' num' : ''), esc(c.label));
    th.addEventListener('click', () => {
      if (sortKey === c.key) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      else { sortKey = c.key; sortDir = c.num ? 'desc' : 'asc'; }
      draw();
      [...trh.children].forEach(x => x.classList.remove('sort-asc', 'sort-desc'));
      th.classList.add(sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
    });
    trh.appendChild(th);
  });
  thead.appendChild(trh); table.appendChild(thead);
  const tbody = el('tbody'); table.appendChild(tbody);
  function draw() {
    let data = rows.slice();
    if (sortKey) {
      const col = cols.find(c => c.key === sortKey);
      data.sort((a, b) => {
        let va = a[sortKey], vb = b[sortKey];
        if (col && col.num) { va = va == null ? -Infinity : +va; vb = vb == null ? -Infinity : +vb; return sortDir === 'asc' ? va - vb : vb - va; }
        va = (va == null ? '' : String(va)).toLowerCase(); vb = (vb == null ? '' : String(vb)).toLowerCase();
        return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      });
    }
    tbody.innerHTML = '';
    data.forEach(r => {
      const tr = el('tr');
      cols.forEach(c => {
        const td = el('td', c.num ? 'num' : '');
        td.innerHTML = c.render ? c.render(r[c.key], r) : esc(r[c.key] == null ? '—' : r[c.key]);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }
  draw();
  wrap.appendChild(table);
  return wrap;
}

function kpi(label, value, delta, cls = '') {
  return `<div class="kpi ${cls}"><div class="kpi-label">${esc(label)}</div>
    <div class="kpi-value" data-count>${esc(value)}</div>${delta ? `<div class="kpi-delta">${esc(delta)}</div>` : ''}</div>`;
}
function pageHead(title, sub) {
  return `<div class="page-head"><div class="page-title">${esc(title)}</div>${sub ? `<div class="page-sub">${esc(sub)}</div>` : ''}</div>`;
}

// ═══════════════════════════════════════════════════════════
// PAGE: TODAY'S GAMES
// ═══════════════════════════════════════════════════════════
async function pageToday() {
  root.innerHTML = pageHead("TODAY'S GAMES") + spinner();
  try {
    const [d, acc] = await Promise.all([api('/api/today'), api('/api/accuracy').catch(() => null)]);
    const dateStr = new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    let html = pageHead("TODAY'S GAMES", dateStr);
    html += heroBand(acc, d.games.length);
    if (d.fallback) html += `<div class="fallback-note">No games scheduled for today — showing the most recent slate (${esc(d.date)}).</div>`;
    if (!d.games.length) { root.innerHTML = html + notice('No predictions available. Run train_and_predict.py to generate.'); return; }
    html += `<div class="count-note"><b>${d.games.length}</b> games predicted</div>`;
    d.games.forEach(g => { html += gameCard(g); });
    root.innerHTML = html;
  } catch (e) { root.innerHTML = pageHead("TODAY'S GAMES") + `<div class="error">${esc(e.message)}</div>`; }
}

function heroBand(acc, gamesToday) {
  if (!acc) return '';
  const wrong = acc.completed - acc.correct;
  const accCol = acc.accuracy >= 60 ? 'var(--green)' : acc.accuracy >= 50 ? 'var(--yellow)' : 'var(--red)';
  const stat = (val, lbl, count, color) =>
    `<div class="hero-stat"><div class="hs-val" ${count ? 'data-count' : ''} style="${color ? `color:${color}` : ''}">${esc(val)}</div><div class="hs-lbl">${esc(lbl)}</div></div>`;
  return `<div class="hero"><div class="hero-inner">
    ${stat(gamesToday, 'Games On Slate', true)}
    <div class="hero-divider"></div>
    ${stat(acc.completed ? acc.accuracy + '%' : '—', 'Season Accuracy', !!acc.completed, accCol)}
    <div class="hero-divider"></div>
    ${stat(`${acc.correct}–${wrong}`, 'Record (W–L)', false)}
    <div class="hero-divider"></div>
    ${stat(acc.total, 'Predictions Made', true)}
    <div class="hero-divider"></div>
    ${stat(acc.cv, 'CV Baseline', false, 'var(--cyan)')}
  </div></div>`;
}

function gameCard(g) {
  const homeWin = g.winner === g.home, awayWin = g.winner === g.away;
  const [hp] = teamPair(g.home), [ap] = teamPair(g.away);
  const badges = [];
  if (g.home_b2b) badges.push(`<span class="badge b2b">🟠 ${esc(g.home)} B2B</span>`);
  if (g.away_b2b) badges.push(`<span class="badge b2b">🟠 ${esc(g.away)} B2B</span>`);
  const sc = s => s.startsWith('W') ? 'streak-w' : s.startsWith('L') ? 'streak-l' : '';
  badges.push(`<span class="badge ${sc(g.away_streak)}"><span class="team-dot" style="color:${ap}"></span>${esc(g.away)} ${esc(g.away_streak)}</span>`);
  badges.push(`<span class="badge ${sc(g.home_streak)}"><span class="team-dot" style="color:${hp}"></span>${esc(g.home)} ${esc(g.home_streak)}</span>`);

  let result = '';
  if (g.actual_winner) {
    const correct = String(g.actual_winner) === g.winner;
    result = `<div class="result-banner ${correct ? 'correct' : 'wrong'}">
      ${correct ? '✅ CORRECT' : '❌ WRONG'} · Final: ${esc(g.away)} ${g.actual_away} — ${g.actual_home} ${esc(g.home)} · Predicted ${esc(g.winner)}</div>`;
  }

  const homePlayers = g.players.filter(p => p.is_home).slice(0, 6);
  const awayPlayers = g.players.filter(p => !p.is_home).slice(0, 6);
  const prow = p => `<div class="proj-player"><span class="proj-name">${esc(p.player)}<span class="pos-tag">${esc(p.pos)}</span></span>
    <span class="proj-probs"><span class="pg">G ${pct0(p.g)}</span><span class="pa">A ${pct0(p.a)}</span><span class="pp">P ${pct0(p.p)}</span></span></div>`;
  const projGrid = g.players.length ? `<hr class="divider" />
    <div class="proj-grid">
      <div class="proj-col"><h4>✈ ${esc(g.away)}</h4>${awayPlayers.map(prow).join('')}</div>
      <div class="proj-col"><h4>🏠 ${esc(g.home)}</h4>${homePlayers.map(prow).join('')}</div>
    </div>` : '';

  const mini = (label, val, good) => `<div class="mini-stat"><div class="ms-label">${label}</div>
    <div class="ms-val ${good ? 'pos' : 'neg'}">${val}</div></div>`;

  return `<div class="card game-card" style="--team:${hp};--team2:${ap}">
    <div class="game-head">
      <div>
        <div class="game-matchup">${esc(g.away)} <span style="color:var(--muted-2)">@</span> ${esc(g.home)}${g.is_ot ? ' <span class="badge">⏱ OT likely</span>' : ''}</div>
        <div class="game-badges">${badges.join('')}</div>
      </div>
      ${confGauge(g.confidence)}
    </div>
    <div class="score-row">
      <div class="score-team ${awayWin ? 'winner' : ''}" style="--tcol:${ap}">
        <div class="st-name">${esc(g.away)}</div><div class="st-score">${g.away_score}</div>
        <div class="st-proj">${g.away_proj.toFixed(1)} proj goals</div>${awayWin ? '<div class="st-win">🏆 PREDICTED WINNER</div>' : ''}
      </div>
      <div class="score-at">@</div>
      <div class="score-team ${homeWin ? 'winner' : ''}" style="--tcol:${hp}">
        <div class="st-name">${esc(g.home)}</div><div class="st-score">${g.home_score}</div>
        <div class="st-proj">${g.home_proj.toFixed(1)} proj goals</div>${homeWin ? '<div class="st-win">🏆 PREDICTED WINNER</div>' : ''}
      </div>
    </div>
    <div class="analytics-row">
      ${mini('HomeIce Diff', signed(g.home_ice, 2), g.home_ice > 0)}
      ${mini('Goal Diff/GP', signed(g.xgf, 2), g.xgf > 0)}
      ${mini('Elo Edge', signed(g.gsax * 100, 0), g.gsax > 0)}
    </div>
    ${result}${projGrid}
  </div>`;
}

// ═══════════════════════════════════════════════════════════
// PAGE: PLAYER PROPS
// ═══════════════════════════════════════════════════════════
let propsData = null;
async function pageProps() {
  root.innerHTML = pageHead('PLAYER PROPS') + spinner();
  try {
    const d = await api('/api/player-props');
    propsData = d;
    if (!d.players.length) { root.innerHTML = pageHead('PLAYER PROPS') + notice('No player projections available.'); return; }
    const teams = ['All Teams', ...new Set(d.players.map(p => p.Team))].sort();
    const poss = ['All Positions', ...new Set(d.players.map(p => p.Pos).filter(Boolean))].sort();
    root.innerHTML = pageHead('PLAYER PROPS', `Projections for ${new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}`) + `
      <div class="filters">
        <div class="field"><label>Team</label><select id="fTeam">${teams.map(t => `<option>${esc(t)}</option>`).join('')}</select></div>
        <div class="field"><label>Position</label><select id="fPos">${poss.map(p => `<option>${esc(p)}</option>`).join('')}</select></div>
        <div class="field"><label>Min Point Probability: <span class="range-val" id="fMinV">20%</span></label>
          <input type="range" id="fMin" min="0" max="80" step="5" value="20" /></div>
      </div>
      <div id="propsCount" class="count-note"></div>
      <div id="propsTable"></div>
      <div class="section-title">TOP 20 BY POINT PROBABILITY</div>
      <div class="chart-box" style="height:420px"><canvas id="propsChart"></canvas></div>`;
    ['fTeam', 'fPos', 'fMin'].forEach(id => document.getElementById(id).addEventListener('input', renderProps));
    renderProps();
  } catch (e) { root.innerHTML = pageHead('PLAYER PROPS') + `<div class="error">${esc(e.message)}</div>`; }
}
function renderProps() {
  const team = document.getElementById('fTeam').value, pos = document.getElementById('fPos').value;
  const min = +document.getElementById('fMin').value; document.getElementById('fMinV').textContent = min + '%';
  let df = propsData.players.filter(p => (team === 'All Teams' || p.Team === team) && (pos === 'All Positions' || p.Pos === pos) && (p.point_prob * 100 >= min));
  document.getElementById('propsCount').innerHTML = `<b>${df.length}</b> players matching filters`;
  const cols = [
    { key: 'Player', label: 'Player' }, { key: 'Pos', label: 'Pos' }, { key: 'Team', label: 'Team' },
    { key: 'is_home', label: 'Loc', render: v => v ? '🏠 Home' : '✈ Away' },
    { key: 'goal_prob', label: 'Goal %', num: true, render: v => pct0(v) },
    { key: 'assist_prob', label: 'Assist %', num: true, render: v => pct0(v) },
    { key: 'point_prob', label: 'Point %', num: true, render: v => pct0(v) },
  ];
  const t = document.getElementById('propsTable'); t.innerHTML = ''; t.appendChild(sortableTable(df, cols, { sortKey: 'point_prob', sortDir: 'desc', tall: true }));

  destroyCharts();
  const top20 = df.slice(0, 20);
  charts.push(new Chart(document.getElementById('propsChart'), {
    type: 'bar',
    data: {
      labels: top20.map(p => p.Player),
      datasets: [
        { label: 'Goal', data: top20.map(p => p.goal_prob), backgroundColor: RED },
        { label: 'Assist', data: top20.map(p => p.assist_prob), backgroundColor: BLUE },
      ],
    },
    options: { ...CHART_DEFAULTS, scales: { x: { stacked: true, ticks: { color: '#94A3B8', maxRotation: 60, minRotation: 45 }, grid: { color: '#1F2937' } }, y: { stacked: true, ticks: { color: '#94A3B8', callback: v => Math.round(v * 100) + '%' }, grid: { color: '#1F2937' } } } },
  }));
}

// ═══════════════════════════════════════════════════════════
// PAGE: MODEL ACCURACY
// ═══════════════════════════════════════════════════════════
async function pageAccuracy() {
  root.innerHTML = pageHead('MODEL ACCURACY') + spinner();
  try {
    const d = await api('/api/accuracy');
    let html = pageHead('MODEL ACCURACY') + `<div class="kpi-row">
      ${kpi('Season Accuracy', d.completed ? d.accuracy + '%' : '—', null, d.accuracy >= 60 ? 'good' : d.accuracy >= 50 ? 'warn' : 'bad')}
      ${kpi('Correct / Completed', `${d.correct} / ${d.completed}`)}
      ${kpi('CV Accuracy (5-fold)', d.cv)}
      ${kpi('Total Predictions', d.total)}
    </div>
    <div class="section-title">ACCURACY BY CONFIDENCE TIER</div>
    <div class="chart-box" style="height:320px"><canvas id="tierChart"></canvas></div>`;
    if (d.rolling.length) html += `<div class="section-title">ROLLING 10-GAME ACCURACY</div>
      <div class="chart-box" style="height:320px"><canvas id="rollChart"></canvas></div>`;
    html += `<div class="section-title">RECENT PREDICTIONS</div><div id="recentTable"></div>`;
    root.innerHTML = html;

    destroyCharts();
    charts.push(new Chart(document.getElementById('tierChart'), {
      type: 'bar',
      data: { labels: d.tiers.map(t => t.tier), datasets: [{ label: 'Accuracy %', data: d.tiers.map(t => t.acc), backgroundColor: d.tiers.map(t => t.acc >= 60 ? GREEN : t.acc >= 50 ? YELLOW : RED) }] },
      options: { ...CHART_DEFAULTS, plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (ctx) => `${d.tiers[ctx.dataIndex].n} games` } } }, scales: { ...CHART_DEFAULTS.scales, y: { min: 0, max: 85, ticks: { color: '#94A3B8' }, grid: { color: '#1F2937' } } } },
    }));
    if (d.rolling.length) {
      charts.push(new Chart(document.getElementById('rollChart'), {
        type: 'line',
        data: { labels: d.rolling.map(r => r.date), datasets: [{ label: 'Rolling accuracy %', data: d.rolling.map(r => r.rolling), borderColor: CY, backgroundColor: 'rgba(0,180,216,.1)', tension: 0.35, fill: true, pointRadius: 0 }] },
        options: { ...CHART_DEFAULTS, scales: { ...CHART_DEFAULTS.scales, y: { min: 30, max: 90, ticks: { color: '#94A3B8' }, grid: { color: '#1F2937' } } } },
      }));
    }
    const cols = [
      { key: 'Date', label: 'Date' }, { key: 'Away', label: 'Away' }, { key: 'Home', label: 'Home' },
      { key: 'Predicted', label: 'Predicted' }, { key: 'Conf', label: 'Conf %', num: true },
      { key: 'Actual', label: 'Actual', render: v => esc(v || '—') },
      { key: 'Result', label: 'Result', render: v => `<span class="chip ${v.toLowerCase()}">${esc(v)}</span>` },
    ];
    document.getElementById('recentTable').appendChild(sortableTable(d.recent, cols, { sortKey: 'Date', sortDir: 'desc', tall: true }));
  } catch (e) { root.innerHTML = pageHead('MODEL ACCURACY') + `<div class="error">${esc(e.message)}</div>`; }
}

// ═══════════════════════════════════════════════════════════
// PAGE: TEAM STATS
// ═══════════════════════════════════════════════════════════
let teamData = null, teamView = 'Standings';
const TEAM_VIEWS = {
  Standings: [['Team', 'Team'], ['GP', 'GP', 1], ['W', 'W', 1], ['L', 'L', 1], ['OTL', 'OTL', 1], ['Points', 'PTS', 1], ['Point%', 'PT%', 1], ['Goal Diff', 'DIFF', 1]],
  Possession: [['Team', 'Team'], ['CF%', 'CF%', 1], ['FF%', 'FF%', 1], ['xGF%', 'xGF%', 1], ['HDCF%', 'HDCF%', 1], ['PDO', 'PDO', 1]],
  Scoring: [['Team', 'Team'], ['GF/G', 'GF/G', 1], ['GA/G', 'GA/G', 1], ['SH%', 'SH%', 1], ['SV%', 'SV%', 1], ['FO%', 'FO%', 1]],
  'Home/Away': [['Team', 'Team'], ['Home W%', 'HOME W%', 1], ['Away W%', 'AWAY W%', 1], ['GF/G', 'GF/G', 1], ['GA/G', 'GA/G', 1]],
};
async function pageTeams() {
  root.innerHTML = pageHead('TEAM STATS') + spinner();
  await loadTeams('2025-26');
}
async function loadTeams(season) {
  try {
    const d = await api('/api/team-stats?season=' + season);
    teamData = d;
    if (!d.teams.length) { root.innerHTML = pageHead('TEAM STATS') + notice('No team stats available.'); return; }
    root.innerHTML = pageHead('TEAM STATS') + `
      <div class="filters">
        <div class="field"><label>Season</label><select id="fSeason"><option ${season === '2025-26' ? 'selected' : ''}>2025-26</option><option ${season === '2024-25' ? 'selected' : ''}>2024-25</option></select></div>
        <div class="field"><label>View</label><div class="seg" id="viewSeg">${Object.keys(TEAM_VIEWS).map(v => `<button class="${v === teamView ? 'active' : ''}" data-v="${v}">${v}</button>`).join('')}</div></div>
      </div>
      <div id="teamTable"></div>
      <div class="section-title">xGF% vs HDCF% — POSSESSION QUALITY</div>
      <div class="chart-box" style="height:480px"><canvas id="teamScatter"></canvas></div>`;
    document.getElementById('fSeason').addEventListener('change', e => loadTeams(e.target.value));
    document.getElementById('viewSeg').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; teamView = b.dataset.v; [...e.currentTarget.children].forEach(x => x.classList.toggle('active', x === b)); renderTeamTable(); });
    renderTeamTable();
    renderTeamScatter();
  } catch (e) { root.innerHTML = pageHead('TEAM STATS') + `<div class="error">${esc(e.message)}</div>`; }
}
function renderTeamTable() {
  const cols = TEAM_VIEWS[teamView].map(([key, label, num]) => ({ key, label, num: !!num, render: key === 'Team' ? teamDot : undefined }));
  const t = document.getElementById('teamTable'); t.innerHTML = '';
  const sortKey = teamView === 'Standings' ? 'Points' : cols[1].key;
  t.appendChild(sortableTable(teamData.teams, cols, { sortKey, sortDir: 'desc', tall: true }));
}
function renderTeamScatter() {
  destroyCharts();
  const pts = teamData.teams.map(t => ({ x: t['xGF%'], y: t['HDCF%'], team: t.Team, r: 4 + (t.Points || 0) / 12 }));
  charts.push(new Chart(document.getElementById('teamScatter'), {
    type: 'scatter',
    data: { datasets: [{ label: 'Teams', data: pts, backgroundColor: 'rgba(0,180,216,.65)', borderColor: CY, pointRadius: pts.map(p => p.r), pointHoverRadius: 9 }] },
    options: { ...CHART_DEFAULTS, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${c.raw.team}: xGF% ${c.raw.x}, HDCF% ${c.raw.y}` } } },
      scales: { x: { title: { display: true, text: 'xGF%', color: '#94A3B8' }, ticks: { color: '#94A3B8' }, grid: { color: '#1F2937' } }, y: { title: { display: true, text: 'HDCF%', color: '#94A3B8' }, ticks: { color: '#94A3B8' }, grid: { color: '#1F2937' } } } },
  }));
}

// ═══════════════════════════════════════════════════════════
// PAGE: HOT & COLD STREAKS
// ═══════════════════════════════════════════════════════════
let streakTab = 'teams';
async function pageStreaks() {
  root.innerHTML = pageHead('HOT & COLD STREAKS', 'Team and player performance over the last 10 games') + `
    <div class="tabs" id="streakTabs">
      <button class="${streakTab === 'teams' ? 'active' : ''}" data-t="teams">Team Streaks</button>
      <button class="${streakTab === 'players' ? 'active' : ''}" data-t="players">Player Streaks</button>
    </div><div id="streakBody">${spinner()}</div>`;
  document.getElementById('streakTabs').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; streakTab = b.dataset.t; [...e.currentTarget.children].forEach(x => x.classList.toggle('active', x === b)); streakTab === 'teams' ? renderTeamStreaks() : renderPlayerStreaks(); });
  streakTab === 'teams' ? renderTeamStreaks() : renderPlayerStreaks();
}
async function renderTeamStreaks() {
  const body = document.getElementById('streakBody'); body.innerHTML = spinner();
  try {
    const d = await api('/api/streaks/teams');
    if (!d.teams.length) { body.innerHTML = notice('No team streak data available.'); return; }
    const hot = d.teams[0], cold = d.teams[d.teams.length - 1];
    const avg = (d.teams.reduce((s, t) => s + t.w10, 0) / d.teams.length).toFixed(1);
    body.innerHTML = `<div class="kpi-row">
      ${kpi('Hottest Team', hot.team, `${hot.streak} · ${hot.record} L10`, 'good')}
      ${kpi('Coldest Team', cold.team, `${cold.streak} · ${cold.record} L10`, 'bad')}
      ${kpi('League Avg Wins (L10)', avg)}
    </div>
    <div class="filters"><div class="field"><label>Filter by streak</label>
      <select id="tierSel"><option>All Teams</option><option>On Fire (8-10W)</option><option>Hot (6-7W)</option><option>Average (5W)</option><option>Cool (4W)</option><option>Cold (0-3W)</option></select></div></div>
    <div id="teamStreakTable"></div>
    <div class="section-title">WINS IN LAST 10 GAMES</div>
    <div class="chart-box" style="height:760px"><canvas id="streakChart"></canvas></div>`;
    const draw = () => {
      const f = document.getElementById('tierSel').value;
      let rows = d.teams;
      if (f.startsWith('On Fire')) rows = rows.filter(t => t.w10 >= 8);
      else if (f.startsWith('Hot')) rows = rows.filter(t => t.w10 >= 6 && t.w10 <= 7);
      else if (f.startsWith('Average')) rows = rows.filter(t => t.w10 === 5);
      else if (f.startsWith('Cool')) rows = rows.filter(t => t.w10 === 4);
      else if (f.startsWith('Cold')) rows = rows.filter(t => t.w10 <= 3);
      const cols = [
        { key: 'team', label: 'Team', render: teamDot }, { key: 'streak', label: 'Streak' },
        { key: 'form', label: 'Form (L10)', render: v => `<span class="${chipClass(v)}">${esc(v)}</span>` },
        { key: 'record', label: 'Record (L10)' }, { key: 'goal_diff', label: 'Goal Diff', num: true },
        { key: 'gf_l10', label: 'GF/G', num: true }, { key: 'ga_l10', label: 'GA/G', num: true },
        { key: 'last_game', label: 'Last Game' },
      ];
      const t = document.getElementById('teamStreakTable'); t.innerHTML = ''; t.appendChild(sortableTable(rows, cols, { sortKey: 'w10', sortDir: 'desc', tall: true }));
    };
    document.getElementById('tierSel').addEventListener('change', draw); draw();
    destroyCharts();
    const sorted = d.teams.slice().sort((a, b) => a.w10 - b.w10);
    charts.push(new Chart(document.getElementById('streakChart'), {
      type: 'bar',
      data: { labels: sorted.map(t => t.team), datasets: [{ label: 'Wins (L10)', data: sorted.map(t => t.w10), backgroundColor: sorted.map(t => t.w10 >= 8 ? RED : t.w10 >= 6 ? GREEN : t.w10 === 5 ? YELLOW : BLUE) }] },
      options: { ...CHART_DEFAULTS, indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 10, ticks: { color: '#94A3B8' }, grid: { color: '#1F2937' } }, y: { ticks: { color: '#94A3B8', font: { size: 10 } }, grid: { display: false } } } },
    }));
  } catch (e) { body.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}
async function renderPlayerStreaks() {
  const body = document.getElementById('streakBody'); body.innerHTML = spinner();
  try {
    const d = await api('/api/streaks/players');
    const teams = ['All Teams', ...d.teams];
    body.innerHTML = `<div class="filters">
      <div class="field"><label>Position</label><select id="pPos"><option>All Skaters</option><option>Forwards</option><option>Defense</option><option>Goalies</option></select></div>
      <div class="field"><label>Team</label><select id="pTeam">${teams.map(t => `<option>${esc(t)}</option>`).join('')}</select></div>
    </div><div id="playerStreakBody">${renderPlayerStreakData(d)}</div>`;
    const reload = async () => {
      const pos = document.getElementById('pPos').value, team = document.getElementById('pTeam').value;
      const pb = document.getElementById('playerStreakBody'); pb.innerHTML = spinner();
      const nd = await api(`/api/streaks/players?pos=${encodeURIComponent(pos)}&team=${encodeURIComponent(team)}`);
      pb.innerHTML = renderPlayerStreakData(nd); mountPlayerStreakTable(nd);
    };
    document.getElementById('pPos').addEventListener('change', reload);
    document.getElementById('pTeam').addEventListener('change', reload);
    mountPlayerStreakTable(d);
  } catch (e) { body.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}
function renderPlayerStreakData(d) {
  if (!d.players.length) return notice('No player gamelog data for the selected filters.');
  const top = d.players[0];
  const avg = (d.players.reduce((s, p) => s + (p.ptspg || 0), 0) / d.players.length).toFixed(2);
  return `<div class="kpi-row">
    ${kpi('Hottest Skater', top.player, `${top.g}G ${top.a}A ${top.pts}PTS L10`, 'good')}
    ${kpi('Avg Pts/G (L10)', avg)}
    ${kpi('Players Shown', d.players.length)}
  </div>
  <div id="playerStreakTable"></div>
  <div class="section-title">TOP 25 SKATERS — POINTS IN LAST 10</div>
  <div class="chart-box" style="height:640px"><canvas id="playerStreakChart"></canvas></div>`;
}
function mountPlayerStreakTable(d) {
  if (!d.players.length) return;
  const cols = [
    { key: 'player', label: 'Player' }, { key: 'pos', label: 'Pos' }, { key: 'team', label: 'Team', render: teamDot },
    { key: 'form', label: 'Streak', render: v => `<span class="${chipClass(v)}">${esc(v)}</span>` },
    { key: 'gp', label: 'GP', num: true }, { key: 'g', label: 'G', num: true }, { key: 'a', label: 'A', num: true },
    { key: 'pts', label: 'PTS', num: true }, { key: 'ptspg', label: 'Pts/G', num: true },
    { key: 'plusminus', label: '+/-', num: true }, { key: 'sog', label: 'SOG', num: true },
  ];
  const t = document.getElementById('playerStreakTable'); t.innerHTML = ''; t.appendChild(sortableTable(d.players, cols, { sortKey: 'pts', sortDir: 'desc', tall: true }));
  destroyCharts();
  const top25 = d.players.slice(0, 25).slice().sort((a, b) => a.pts - b.pts);
  charts.push(new Chart(document.getElementById('playerStreakChart'), {
    type: 'bar',
    data: { labels: top25.map(p => `${p.player} (${(p.team || '').slice(0, 3)})`), datasets: [{ label: 'Points (L10)', data: top25.map(p => p.pts), backgroundColor: top25.map(p => p.pts >= 12 ? RED : p.pts >= 8 ? GREEN : p.pts >= 5 ? YELLOW : BLUE) }] },
    options: { ...CHART_DEFAULTS, indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#94A3B8' }, grid: { color: '#1F2937' } }, y: { ticks: { color: '#94A3B8', font: { size: 10 } }, grid: { display: false } } } },
  }));
}

// ═══════════════════════════════════════════════════════════
// PAGE: PLAYOFFS
// ═══════════════════════════════════════════════════════════
async function pagePlayoffs() {
  root.innerHTML = pageHead('2025–26 NHL PLAYOFFS', 'First Round — Live Series Tracker & Predictions') + spinner();
  try {
    const d = await api('/api/playoffs');
    if (!d.series.length) { root.innerHTML = pageHead('2025–26 NHL PLAYOFFS') + notice('No 2025–26 playoff data found.'); return; }
    let html = pageHead('2025–26 NHL PLAYOFFS', 'First Round — Live Series Tracker & Predictions') + '<div class="section-title">FIRST ROUND SERIES</div><div class="series-grid">';
    d.series.forEach(s => { html += seriesCard(s); });
    html += '</div><div class="section-title">SERIES WINNER PREDICTIONS</div><div class="proj-grid">';
    html += `<div><h4 style="color:var(--cyan);margin-bottom:8px">EASTERN CONFERENCE</h4><div id="predEast"></div></div>`;
    html += `<div><h4 style="color:var(--cyan);margin-bottom:8px">WESTERN CONFERENCE</h4><div id="predWest"></div></div>`;
    html += `</div><p class="page-sub" style="margin-top:20px">Series win probability uses a Markov model seeded with regular-season Point%, xGF%, and home-ice advantage (≈+4%). Updates automatically as results load.</p>`;
    root.innerHTML = html;
    mountPredTable('predEast', d.east);
    mountPredTable('predWest', d.west);
  } catch (e) { root.innerHTML = pageHead('2025–26 NHL PLAYOFFS') + `<div class="error">${esc(e.message)}</div>`; }
}
function seriesCard(s) {
  const hiW = s.hi_wins, oppW = s.opp_wins;
  let lead, badge;
  if (s.series_over) { lead = hiW === 4 ? `${s.home_ice_name} wins ${hiW}–${oppW}` : `${s.opp_name} wins ${oppW}–${hiW}`; badge = '🏆'; }
  else if (hiW > oppW) { lead = `${s.home_ice_name} leads ${hiW}–${oppW}`; badge = '🟢'; }
  else if (oppW > hiW) { lead = `${s.opp_name} leads ${oppW}–${hiW}`; badge = '🟡'; }
  else { lead = `Series tied ${hiW}–${oppW}`; badge = '⚪'; }
  const [hiCol] = teamPair(s.home_ice_name), [oppCol] = teamPair(s.opp_name);
  const bar = (p, over, won, col) => over
    ? `<div class="series-lead">${won ? '🏆 Winner' : 'Eliminated'}</div>`
    : `<div class="prob-bar"><span data-w="${Math.round(p * 100)}%" style="background:linear-gradient(90deg, ${col}, color-mix(in srgb, ${col} 55%, #ffffff))"></span></div><div class="series-lead">Predicted series win: <b>${Math.round(p * 100)}%</b></div>`;
  const games = s.games.map(g => ({ ...g }));
  const cols = [
    { key: 'game', label: 'Game' }, { key: 'date', label: 'Date' }, { key: 'away', label: 'Away' },
    { key: 'home', label: 'Home' }, { key: 'score', label: 'Score' },
    { key: 'winner', label: 'Winner', render: v => v === 'Pending' ? '<span class="chip pending">Pending</span>' : esc(v) },
  ];
  const tblId = 'gl' + Math.random().toString(36).slice(2, 8);
  setTimeout(() => { const holder = document.getElementById(tblId); if (holder) holder.appendChild(sortableTable(games, cols)); }, 0);
  return `<div class="card" style="--team:${hiCol};--team2:${oppCol}">
    <div class="series-head"><div class="series-title">${badge} ${esc(s.home_ice_name)} vs ${esc(s.opp_name)}</div><div class="series-lead">${esc(lead)}</div></div>
    <div class="series-teams">
      <div class="series-team" style="--tcol:${hiCol}"><div class="stt-name"><span class="team-dot" style="color:${hiCol}"></span>${esc(s.home_ice_name)}</div><div class="stt-wins">${hiW}</div>${bar(s.win_prob_hi, s.series_over, hiW === 4, hiCol)}</div>
      <div class="series-vs">VS</div>
      <div class="series-team" style="--tcol:${oppCol}"><div class="stt-name"><span class="team-dot" style="color:${oppCol}"></span>${esc(s.opp_name)}</div><div class="stt-wins">${oppW}</div>${bar(s.win_prob_opp, s.series_over, oppW === 4, oppCol)}</div>
    </div>
    <hr class="divider" /><div style="font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Game Log</div>
    <div id="${tblId}"></div>
  </div>`;
}
function mountPredTable(id, rows) {
  const holder = document.getElementById(id);
  if (!holder) return;
  if (!rows.length) { holder.innerHTML = notice('No series in this conference.'); return; }
  const cols = [{ key: 'matchup', label: 'Matchup' }, { key: 'winner', label: 'Predicted Winner' }, { key: 'confidence', label: 'Confidence' }, { key: 'series', label: 'Series' }];
  holder.appendChild(sortableTable(rows, cols));
}

// ═══════════════════════════════════════════════════════════
// PAGE: STATS GUIDE
// ═══════════════════════════════════════════════════════════
const GUIDE = {
  'Possession & Shot Quality': [
    ['CF% — Corsi For Percentage', 'The percentage of all shot attempts (goals, shots on goal, missed shots, and blocked shots) taken by a team at 5-on-5.', 'CF% = Attempts For / (Attempts For + Attempts Against)', 'Above 50% means the team generates more shot attempts than they allow — the broadest measure of possession. League average is 50%.'],
    ['FF% — Fenwick For Percentage', 'Same as CF% but excludes blocked shots. Unblocked attempts are a better proxy for shot quality.', 'FF% = Unblocked For / (Unblocked For + Unblocked Against)', 'Tracks closely with CF%, but shot-blocking defenses widen the gap.'],
    ['xGF% — Expected Goals For Percentage', 'Percentage of expected goals generated, weighting each shot by its probability of scoring (type, location, situation).', 'xGF% = xGoals For / (xGoals For + xGoals Against)', 'The most predictive possession stat. Above 50% signals genuine shot-quality dominance. (Shown as team context — the game model now uses point-in-time Elo and goal-based form rather than season-aggregate xGF%.)'],
    ['HDCF% — High-Danger Corsi For %', 'Corsi restricted to high-danger attempts — shots from the slot, roughly within 20 feet of the net.', 'HDCF% = HD Attempts For / (HD For + HD Against)', 'High-danger chances convert far more often. Pairs with xGF% to tell the full possession story.'],
    ['PDO', "Sum of a team's 5-on-5 save percentage and shooting percentage.", 'PDO = SV% + SH% (e.g. .923 + .080 = 100.3)', 'Regresses toward 100. Above 102 = likely lucky; below 98 = due for better results. Used as a luck/sustainability signal.'],
  ],
  'Goalie & Shooting': [
    ['SV% — Save Percentage', 'Fraction of shots on goal a goalie stops.', 'SV% = Saves / Shots On Goal Against', 'League average ~.906–.912. Above .920 over a full season is elite. Shown here as a percentage.'],
    ['SH% — Shooting Percentage', 'Percentage of shots on goal that become goals.', 'SH% = Goals / Shots On Goal', 'League average ~8–10%. Regresses toward the mean; sustained highs indicate elite skaters or puck luck.'],
    ['GSAX — Goals Saved Above Expected', 'How many more goals a goalie prevented vs an average goalie facing the same shots.', 'GSAX = Expected Goals Against − Actual Goals Against', 'Positive = outperformed. +10 over a season is elite. (Shown as team context — per-game goalie data isn\'t captured, so the game model doesn\'t use GSAX directly.)'],
    ['GF/G — Goals For Per Game', 'Average goals scored per game (all situations).', 'GF/G = Total Goals For / Games Played', 'Top teams sit above 3.5. Pair with GA/G for goal differential.'],
    ['GA/G — Goals Against Per Game', 'Average goals allowed per game.', 'GA/G = Total Goals Against / Games Played', 'Elite defenses allow under 2.5. GF/G − GA/G is one of the strongest predictors of success.'],
    ['FO% — Faceoff Win Percentage', 'Percentage of faceoffs won.', 'FO% = Faceoffs Won / Total Faceoffs', 'League average 50%. Modest but real effect on outcomes, especially defensive-zone draws.'],
  ],
  'Model Inputs': [
    ['Goal Differential / Game', "Difference between the two teams' season-to-date goal differential per game, computed point-in-time (only games before the matchup).", 'Goal Diff/GP = Home (GF−GA)/game − Away (GF−GA)/game', 'Positive = home team has outscored opponents by more this season. One of the highest-weight model features.'],
    ['Elo Edge', "Gap in the two teams' Elo ratings. Elo updates game-by-game (home-ice + margin-of-victory) and carries across seasons, so it is leak-free.", 'Elo Edge = Home Elo − Away Elo (before home-ice adjustment)', "Positive = home team is the stronger side by Elo. Elo and its win probability are the model's most important features."],
    ['HomeIce Differential', "Composite home-ice score combining the home team's point-in-time home win rate and the away team's road win rate.", 'HomeIce Diff = (Home home-W% − Away road-W%) × 6', 'Adjusts the baseline per matchup — some arenas confer a bigger edge than others.'],
    ['Back-to-Back Flag', 'Whether a team is playing its second game in two nights.', 'B2B = 1 if a game was played the previous day', 'B2B meaningfully lowers win probability, especially with travel. Applied separately for home/away.'],
    ['Model Confidence %', 'Ensemble probability that the predicted winner actually wins.', 'Confidence = max(P(home win), P(away win))', '≥65% High (green) · 55–65% Medium (yellow) · <55% Low (red). Historical CV accuracy 60.5%.'],
  ],
};
let guideTab = 'Possession & Shot Quality';
function pageGuide() {
  destroyCharts();
  const tabs = Object.keys(GUIDE);
  let html = pageHead('STATS GUIDE', 'Definitions for every stat used in this model and dashboard') + `<div class="tabs" id="guideTabs">${tabs.map(t => `<button class="${t === guideTab ? 'active' : ''}" data-t="${esc(t)}">${esc(t)}</button>`).join('')}</div><div id="guideBody"></div>`;
  root.innerHTML = html;
  const draw = () => {
    const items = GUIDE[guideTab];
    let b = '<div class="accordion">' + items.map(([name, what, formula, how]) => `<details><summary>${esc(name)}</summary><div class="body"><b>What it measures:</b> ${esc(what)}<code>${esc(formula)}</code><b>How to read it:</b> ${esc(how)}</div></details>`).join('') + '</div>';
    if (guideTab === 'Model Inputs') {
      b += `<div class="section-title">DATA SOURCES</div><div class="guide-sources">
        <div class="card"><h4>Natural Stat Trick</h4><ul><li>CF%, FF%, xGF%, HDCF%</li><li>GSAX, PDO</li><li>All 5-on-5 splits</li></ul></div>
        <div class="card"><h4>NHL Official API</h4><ul><li>Game schedules &amp; scores</li><li>Roster &amp; player data</li><li>Starting goalies &amp; lineups</li></ul></div>
        <div class="card"><h4>Hockey Reference</h4><ul><li>Historical game stats</li><li>Team standings</li><li>Season aggregates</li></ul></div></div>`;
    }
    document.getElementById('guideBody').innerHTML = b;
  };
  document.getElementById('guideTabs').addEventListener('click', e => { const btn = e.target.closest('button'); if (!btn) return; guideTab = btn.dataset.t; [...e.currentTarget.children].forEach(x => x.classList.toggle('active', x === btn)); draw(); });
  draw();
}

// ═══════════════════════════════════════════════════════════
// Router + status
// ═══════════════════════════════════════════════════════════
const PAGES = { today: pageToday, playoffs: pagePlayoffs, props: pageProps, accuracy: pageAccuracy, teams: pageTeams, streaks: pageStreaks, guide: pageGuide };
let currentPage = 'today';
function go(page) {
  currentPage = page;
  destroyCharts();
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.page === page));
  Promise.resolve((PAGES[page] || pageToday)()).then(postRender).catch(() => {});
  if (window.innerWidth <= 900) closeNav();
  history.replaceState(null, '', '#' + page);
}
document.getElementById('nav').addEventListener('click', e => { const b = e.target.closest('.nav-item'); if (b) go(b.dataset.page); });

// mobile nav
const sidebar = document.getElementById('sidebar'), scrim = document.getElementById('scrim');
function openNav() { sidebar.classList.add('open'); scrim.classList.add('show'); }
function closeNav() { sidebar.classList.remove('open'); scrim.classList.remove('show'); }
document.getElementById('navToggle').addEventListener('click', openNav);
scrim.addEventListener('click', closeNav);

async function loadStatus() {
  try {
    const s = await api('/api/status');
    document.getElementById('stLastRun').textContent = s.last_run || '—';
    document.getElementById('stAcc').textContent = s.season_accuracy || '—';
    document.getElementById('stTotal').textContent = s.total ?? '—';
    if (s.refreshed_at) document.getElementById('stUpdated').textContent = 'Data as of ' + s.refreshed_at;
  } catch { /* ignore */ }
}

const refreshBtn = document.getElementById('refreshBtn');
refreshBtn.addEventListener('click', async () => {
  if (refreshBtn.disabled) return;
  refreshBtn.disabled = true;
  refreshBtn.classList.add('spinning');
  const label = refreshBtn.querySelector('.rb-txt');
  const prev = label.textContent;
  label.textContent = 'Refreshing…';
  try {
    await fetch('/api/refresh', { method: 'POST' });
    await loadStatus();
    go(currentPage);            // re-render the active page with fresh data
    label.textContent = 'Refreshed ✓';
    toast('✓ Data refreshed from MotherDuck');
  } catch {
    label.textContent = 'Failed — retry';
    toast('✕ Refresh failed — try again');
  } finally {
    refreshBtn.classList.remove('spinning');
    refreshBtn.disabled = false;
    setTimeout(() => { label.textContent = prev; }, 1600);
  }
});

loadStatus();
go((location.hash || '#today').slice(1));

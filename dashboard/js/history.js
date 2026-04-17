/**
 * tradIA Account History — history.js
 * =====================================
 * Fetches authenticated Binance account trade history via the
 * dashboard.py read-only API. Does NOT call any trading logic.
 */

/* ================================================================
   Utilities
   ================================================================ */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function escHtml(str) {
  const d = document.createElement('div');
  d.textContent = String(str ?? '');
  return d.innerHTML;
}

function fmt(n, dec = 2) {
  const num = parseFloat(n);
  if (isNaN(num)) return '—';
  return num.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function fmtUsdt(n, alwaysSign = false) {
  const num = parseFloat(n);
  if (isNaN(num)) return '—';
  const sign = alwaysSign ? (num >= 0 ? '+' : '-') : (num < 0 ? '-' : '');
  return `${sign}$${fmt(Math.abs(num), 2)}`;
}

function fmtTime(ms) {
  if (!ms) return '—';
  return new Date(ms).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
}

async function apiFetch(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
  return r.json();
}

function showToast(message, type = 'info', duration = 3500) {
  const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ'}</span>
                  <span class="toast-msg">${escHtml(message)}</span>`;
  const container = document.getElementById('toast-container');
  if (container) container.appendChild(el);
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('visible')));
  setTimeout(() => {
    el.classList.remove('visible');
    setTimeout(() => el.remove(), 350);
  }, duration);
}

/* ================================================================
   State
   ================================================================ */
const H = {
  trades:         [],
  filtered:       [],
  income:         [],          // raw income entries from /api/account/income
  liquidationIds: new Set(),   // trade IDs excluded from both metrics and table
  sortKey:        'time',
  sortDir:        'desc',
  page:           1,
  pageSize:       25,
  filter:         'ALL',   // ALL | BUY | SELL | CLOSE
  search:         '',
  curveChart:     null,
  monthlyChart:   null,
};

/* ================================================================
   Clock
   ================================================================ */
function updateClock() {
  const el = $('#nav-clock');
  if (el) el.textContent = new Date().toUTCString().slice(0, 25).replace('GMT', 'UTC');
}

/* ================================================================
   Error banner
   ================================================================ */
function showError(msg) {
  const el = $('#error-banner');
  if (!el) return;
  if (msg) { el.textContent = `Binance API error: ${escHtml(msg)}`; el.style.display = 'block'; }
  else      { el.style.display = 'none'; }
}

/* ================================================================
   Summary cards
   ================================================================ */
function setCard(id, text, color) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  if (color) el.style.color = color;
}

function renderSummary(s) {
  const net = s.net_profit ?? 0;
  setCard('#s-net-profit',    fmtUsdt(net, true),
    net >= 0 ? 'var(--green)' : 'var(--red)');

  const rPnl = s.total_realized_pnl ?? 0;
  setCard('#s-realized-pnl', fmtUsdt(rPnl, true),
    rPnl >= 0 ? 'var(--green)' : 'var(--red)');

  const fees = Math.abs(s.total_commission ?? 0);
  setCard('#s-total-fees', `-$${fmt(fees, 2)}`, 'var(--red)');

  const fund = s.total_funding ?? 0;
  setCard('#s-funding', fmtUsdt(fund, true),
    fund >= 0 ? 'var(--green)' : 'var(--red)');

  setCard('#s-winrate', ((s.win_rate ?? 0) * 100).toFixed(1) + '%');
  const wlEl = $('#s-wins-losses');
  if (wlEl) wlEl.textContent = `${s.wins ?? 0} W / ${s.losses ?? 0} L`;

  setCard('#s-closed', String(s.total_closed_trades ?? 0));
  const fillsEl = $('#s-fills-sub');
  if (fillsEl) fillsEl.textContent = `${s.total_fills ?? 0} fills total`;

  const aw = s.avg_win ?? 0;
  setCard('#s-avg-win',  `+$${fmt(aw,  2)}`, 'var(--green)');

  const al = s.avg_loss ?? 0;
  setCard('#s-avg-loss', `-$${fmt(Math.abs(al), 2)}`, 'var(--red)');

  setCard('#s-rr', (s.rr_ratio ?? 0).toFixed(2) + 'R');

  const syncEl = $('#sync-time');
  if (syncEl) syncEl.textContent = s.synced_at ?? '—';

  if (s.error) showError(s.error);
  else         showError(null);
}

/* ================================================================
   Cumulative PnL chart (area)
   ================================================================ */
function renderCurveChart(curve) {
  const canvas = $('#curve-canvas');
  if (!canvas) return;

  if (H.curveChart) { H.curveChart.destroy(); H.curveChart = null; }

  if (!curve || !curve.length) {
    renderEmptyChart(canvas, 'No closed positions yet');
    return;
  }

  const finalVal  = curve[curve.length - 1].value ?? 0;
  const positive  = finalVal >= 0;
  const lineColor = positive ? '#0ecb81' : '#f6465d';
  const fillColor = positive ? 'rgba(14,203,129,0.08)' : 'rgba(246,70,93,0.08)';

  H.curveChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels:   curve.map(p => new Date(p.time * 1000)),
      datasets: [{
        label:           'Cumulative PnL (USDT)',
        data:            curve.map(p => p.value),
        borderColor:     lineColor,
        backgroundColor: fillColor,
        borderWidth:     2,
        fill:            true,
        tension:         0.3,
        pointRadius:     curve.length > 60 ? 0 : 3,
        pointHoverRadius: 5,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => fmtTime(items[0].parsed.x),
            label: (item)  => ` PnL: ${item.parsed.y >= 0 ? '+' : ''}$${item.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          type: 'time',
          time: { tooltipFormat: 'yyyy-MM-dd HH:mm' },
          ticks: { color: '#4a5568', maxTicksLimit: 6, font: { size: 10 } },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          ticks: {
            color: '#4a5568',
            font:  { size: 10 },
            callback: (v) => `$${v >= 0 ? '' : ''}${v.toFixed(2)}`,
          },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
      },
    },
  });
}

/* ================================================================
   Monthly PnL bar chart
   ================================================================ */
function renderMonthlyChart(monthly) {
  const canvas = $('#monthly-canvas');
  if (!canvas) return;

  if (H.monthlyChart) { H.monthlyChart.destroy(); H.monthlyChart = null; }

  if (!monthly || !monthly.length) {
    renderEmptyChart(canvas, 'No monthly data');
    return;
  }

  const values  = monthly.map(m => m.pnl);
  const colors  = values.map(v => v >= 0 ? 'rgba(14,203,129,0.65)' : 'rgba(246,70,93,0.65)');
  const borders = values.map(v => v >= 0 ? '#0ecb81' : '#f6465d');

  H.monthlyChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels:   monthly.map(m => m.month),
      datasets: [{
        label:           'Monthly PnL (USDT)',
        data:            values,
        backgroundColor: colors,
        borderColor:     borders,
        borderWidth:     1,
        borderRadius:    3,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => ` ${item.parsed.y >= 0 ? '+' : ''}$${item.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#4a5568', font: { size: 10 } },
          grid:  { display: false },
        },
        y: {
          ticks: {
            color: '#4a5568',
            font:  { size: 10 },
            callback: (v) => `$${v.toFixed(0)}`,
          },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
      },
    },
  });
}

function renderEmptyChart(canvas, message) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#4a5568';
  ctx.font      = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(message, canvas.width / 2, canvas.height / 2 + 40);
}

/* ================================================================
   Table — filtering, sorting, pagination
   ================================================================ */
function applyFilters() {
  let rows = H.trades.filter(r => !H.liquidationIds.has(String(r.id)));

  if (H.filter === 'BUY')   rows = rows.filter(r => r.side === 'BUY');
  if (H.filter === 'SELL')  rows = rows.filter(r => r.side === 'SELL');
  if (H.filter === 'CLOSE') rows = rows.filter(r => r.realizedPnl !== 0);

  if (H.search) {
    const q = H.search.toLowerCase();
    rows = rows.filter(r =>
      fmtTime(r.time).toLowerCase().includes(q) ||
      String(r.side).toLowerCase().includes(q)  ||
      String(r.price).includes(q)               ||
      String(r.realizedPnl).includes(q)
    );
  }

  rows.sort((a, b) => {
    let av = H.sortKey === 'net'
      ? (a.realizedPnl - a.commission)
      : (a[H.sortKey] ?? 0);
    let bv = H.sortKey === 'net'
      ? (b.realizedPnl - b.commission)
      : (b[H.sortKey] ?? 0);

    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    if (av < bv) return H.sortDir === 'asc' ? -1 : 1;
    if (av > bv) return H.sortDir === 'asc' ?  1 : -1;
    return 0;
  });

  H.filtered = rows;
  H.page     = 1;
}

function renderTable() {
  applyFilters();

  const tbody   = $('#history-tbody');
  const countEl = $('#history-count');
  const total   = H.filtered.length;

  if (countEl) countEl.textContent = `${total.toLocaleString()} fills`;

  if (!total) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;padding:40px;color:var(--text-3)">
      ${H.trades.length
        ? 'No fills match the current filter'
        : 'No trade history found — click Sync Now to fetch from Binance'}
    </td></tr>`;
    renderPagination(total);
    return;
  }

  const start = (H.page - 1) * H.pageSize;
  const page  = H.filtered.slice(start, start + H.pageSize);

  tbody.innerHTML = page.map(t => {
    const net       = t.realizedPnl - t.commission;
    const isClose   = t.realizedPnl !== 0;
    const sideClass = t.side === 'BUY' ? 'buy' : 'sell';
    const pnlColor  = t.realizedPnl >= 0 ? 'var(--green)' : 'var(--red)';
    const netColor  = net >= 0           ? 'var(--green)' : 'var(--red)';

    const typeBadge = isClose
      ? `<span class="badge badge-green" style="font-size:9px">CLOSE</span>`
      : `<span class="badge" style="font-size:9px;background:rgba(100,116,139,.15);color:var(--text-3)">OPEN</span>`;

    const roleBadge = t.maker
      ? `<span class="badge badge-blue" style="font-size:9px">MAKER</span>`
      : `<span class="badge badge-yellow" style="font-size:9px">TAKER</span>`;

    const pnlCell = isClose
      ? `<span style="color:${pnlColor}">${fmtUsdt(t.realizedPnl, true)}</span>`
      : `<span style="color:var(--text-3)">—</span>`;

    const netCell = isClose
      ? `<span style="color:${netColor}">${fmtUsdt(net, true)}</span>`
      : `<span style="color:var(--red)">-$${escHtml(fmt(t.commission, 4))}</span>`;

    return `<tr>
      <td class="mono" style="font-size:10px;color:var(--text-3)">${escHtml(fmtTime(t.time))}</td>
      <td><span class="dir-badge ${sideClass}">${escHtml(t.side)}</span></td>
      <td class="mono">$${escHtml(fmt(t.price, 2))}</td>
      <td class="mono">${escHtml(fmt(t.qty, 4))}</td>
      <td class="mono">$${escHtml(fmt(t.quoteQty, 2))}</td>
      <td class="mono">${pnlCell}</td>
      <td class="mono" style="color:var(--red)">-$${escHtml(fmt(t.commission, 4))}</td>
      <td class="mono">${netCell}</td>
      <td>${typeBadge}</td>
      <td>${roleBadge}</td>
    </tr>`;
  }).join('');

  // Sync sort arrows
  $$('[data-hsort]').forEach(th => {
    const arrow = th.querySelector('.sort-arrow');
    if (!arrow) return;
    if (th.dataset.hsort === H.sortKey) {
      arrow.textContent = H.sortDir === 'asc' ? '↑' : '↓';
      arrow.style.color = 'var(--accent)';
    } else {
      arrow.textContent = '↕';
      arrow.style.color = '';
    }
  });

  renderPagination(total);
}

function renderPagination(total) {
  const pg    = $('#history-pagination');
  if (!pg) return;
  const pages = Math.ceil(total / H.pageSize);
  if (pages <= 1) { pg.innerHTML = ''; return; }

  const prev = H.page > 1;
  const next = H.page < pages;
  let html = `<button class="page-btn" data-hpage="${H.page - 1}" ${prev ? '' : 'disabled'}>&#8249; Prev</button>`;

  for (let i = 1; i <= pages; i++) {
    if (i === 1 || i === pages || Math.abs(i - H.page) <= 2) {
      html += `<button class="page-btn ${i === H.page ? 'active' : ''}" data-hpage="${i}">${i}</button>`;
    } else if (Math.abs(i - H.page) === 3) {
      html += `<span style="color:var(--text-3);padding:0 4px">…</span>`;
    }
  }
  html += `<button class="page-btn" data-hpage="${H.page + 1}" ${next ? '' : 'disabled'}>Next &#8250;</button>`;
  pg.innerHTML = html;
}

/* ================================================================
   Data loading
   ================================================================ */
async function loadSummary() {
  try {
    const raw = await apiFetch('/api/account/summary');
    // Re-derive metrics from income entries so liquidations are excluded.
    // Fall back to the pre-computed backend summary when income is unavailable.
    H.liquidationIds = new Set();   // reset before each (re)computation
    const s = H.income.length
      ? computeFilteredSummary(H.income, raw)
      : raw;
    renderSummary(s);
    renderCurveChart(s.curve    || []);
    renderMonthlyChart(s.monthly || []);
  } catch (e) {
    showError(e.message);
    console.error('loadSummary:', e);
  }
}

async function loadTrades() {
  try {
    H.trades = await apiFetch('/api/account/trades');
    renderTable();
  } catch (e) {
    const tbody = $('#history-tbody');
    if (tbody) tbody.innerHTML = `<tr><td colspan="10"
      style="text-align:center;padding:40px;color:var(--red)">
      Failed to load fills: ${escHtml(e.message)}
    </td></tr>`;
    console.error('loadTrades:', e);
  }
}

async function loadIncome() {
  try {
    H.income = await apiFetch('/api/account/income');
  } catch (e) {
    console.warn('loadIncome failed — liquidation filter inactive:', e);
    H.income = [];
  }
}

/**
 * Recompute all summary metrics from raw income entries, excluding any
 * REALIZED_PNL entry whose Binance `info` field equals "LIQUIDATION".
 * Commission and funding entries are always included (they reduce net profit
 * regardless of how the position was closed).
 * Non-metric fields (total_fills, synced_at, error) are taken from the
 * pre-computed backend summary so we never need to re-fetch trades.
 */
function computeFilteredSummary(income, backendSummary) {
  // Live exchange: Binance sets info = "LIQUIDATION" on liquidated positions.
  const isExplicitLiquidation = (e) =>
    e.incomeType === 'REALIZED_PNL' &&
    String(e.info || '').toUpperCase() === 'LIQUIDATION';

  const hasExplicit = income.some(isExplicitLiquidation);

  // Testnet fallback: demo-fapi.binance.com puts the trade ID in `info`
  // instead of "LIQUIDATION", so explicit detection misses them.
  // Detect the single most extreme outlier loss: flag it when it is both
  // ≥ $200 AND at least 3× larger in absolute value than the next-worst loss.
  if (!hasExplicit) {
    const byLoss = income
      .filter(e => e.incomeType === 'REALIZED_PNL')
      .map(e => ({ val: parseFloat(e.income) || 0, id: String(e.tradeId || e.tranId) }))
      .sort((a, b) => a.val - b.val);          // most negative first

    if (byLoss.length >= 2) {
      const worst  = Math.abs(byLoss[0].val);
      const second = Math.abs(byLoss[1].val);
      if (worst >= 200 && second > 0 && worst / second >= 3) {
        H.liquidationIds.add(byLoss[0].id);
      }
    }
  } else {
    income
      .filter(e => isExplicitLiquidation(e))
      .forEach(e => H.liquidationIds.add(String(e.tradeId || e.tranId)));
  }

  const isLiquidation = (e) =>
    isExplicitLiquidation(e) ||
    (e.incomeType === 'REALIZED_PNL' && H.liquidationIds.has(String(e.tradeId || e.tranId)));

  const pnlEntries  = income.filter(e => e.incomeType === 'REALIZED_PNL' && !isLiquidation(e));
  const commEntries = income.filter(e => e.incomeType === 'COMMISSION');
  const fundEntries = income.filter(e => e.incomeType === 'FUNDING_FEE');

  const pnlVals  = pnlEntries.map(e => parseFloat(e.income) || 0);
  const wins     = pnlVals.filter(v => v > 0);
  const losses   = pnlVals.filter(v => v < 0);
  const closed   = wins.length + losses.length;

  const totalPnl  = pnlVals.reduce((a, b) => a + b, 0);
  const totalComm = commEntries.reduce((a, e) => a + (parseFloat(e.income) || 0), 0);
  const totalFund = fundEntries.reduce((a, e) => a + (parseFloat(e.income) || 0), 0);
  const netProfit = totalPnl + totalComm + totalFund;

  const grossPnl  = wins.reduce((a, b) => a + b, 0);
  const grossLoss = losses.reduce((a, b) => a + b, 0);
  const winRate   = closed ? wins.length / closed : 0;
  const avgWin    = wins.length   ? grossPnl  / wins.length   : 0;
  const avgLoss   = losses.length ? grossLoss / losses.length : 0;
  const rr        = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0;

  // Cumulative PnL curve — one point per non-liquidation closed position
  const sortedPnl = [...pnlEntries].sort((a, b) => (a.time || 0) - (b.time || 0));
  let cum = 0;
  const curve = sortedPnl.map(e => {
    cum += parseFloat(e.income) || 0;
    return { time: Math.floor(e.time / 1000), value: Math.round(cum * 10000) / 10000 };
  });

  // Monthly breakdown — liquidations excluded
  const monthly = {};
  for (const e of pnlEntries) {
    const dt  = new Date(e.time);
    const key = `${dt.getUTCFullYear()}-${String(dt.getUTCMonth() + 1).padStart(2, '0')}`;
    monthly[key] = (monthly[key] || 0) + (parseFloat(e.income) || 0);
  }
  const monthlyList = Object.entries(monthly)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([month, pnl]) => ({ month, pnl: Math.round(pnl * 10000) / 10000 }));

  const r = (v) => Math.round(v * 10000) / 10000;
  return {
    // Pass-through fields from backend (not recomputed here)
    total_fills:         backendSummary.total_fills,
    synced_at:           backendSummary.synced_at,
    error:               backendSummary.error,
    // Recomputed fields (liquidations excluded)
    total_closed_trades: closed,
    wins:                wins.length,
    losses:              losses.length,
    win_rate:            r(winRate),
    gross_pnl:           r(grossPnl),
    gross_loss:          r(grossLoss),
    total_realized_pnl:  r(totalPnl),
    total_commission:    r(totalComm),
    total_funding:       r(totalFund),
    net_profit:          r(netProfit),
    avg_win:             r(avgWin),
    avg_loss:            r(avgLoss),
    rr_ratio:            r(rr),
    curve,
    monthly:             monthlyList,
  };
}

async function syncNow() {
  const btn = $('#sync-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Syncing…'; }

  try {
    const r    = await fetch('/api/account/sync', { method: 'POST' });
    const data = await r.json();
    if (data.error) {
      showToast(`Sync error: ${data.error}`, 'error', 5000);
    } else {
      showToast('Sync complete — data refreshed', 'success');
      await loadIncome();
      await Promise.all([loadSummary(), loadTrades()]);
    }
  } catch (e) {
    showToast(`Sync failed: ${e.message}`, 'error', 5000);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '↻ Sync Now'; }
  }
}

/* ================================================================
   Event listeners
   ================================================================ */
function bindEvents() {
  // Sort column headers
  $$('[data-hsort]').forEach(th => {
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {
      const key = th.dataset.hsort;
      if (H.sortKey === key) {
        H.sortDir = H.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        H.sortKey = key;
        H.sortDir = 'desc';
      }
      renderTable();
    });
  });

  // Filter buttons
  $$('[data-hfilter]').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('[data-hfilter]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      H.filter = btn.dataset.hfilter;
      renderTable();
    });
  });

  // Search
  const searchEl = $('#history-search');
  if (searchEl) {
    let debounce;
    searchEl.addEventListener('input', () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => {
        H.search = searchEl.value.trim();
        renderTable();
      }, 200);
    });
  }

  // Pagination (event delegation)
  const pgEl = $('#history-pagination');
  if (pgEl) {
    pgEl.addEventListener('click', e => {
      const btn = e.target.closest('[data-hpage]');
      if (!btn || btn.disabled) return;
      H.page = parseInt(btn.dataset.hpage, 10);
      renderTable();
      window.scrollTo({ top: document.getElementById('history-pagination').offsetTop - 80, behavior: 'smooth' });
    });
  }

  // Sync button
  const syncBtn = $('#sync-btn');
  if (syncBtn) syncBtn.addEventListener('click', syncNow);
}

/* ================================================================
   Init
   ================================================================ */
async function init() {
  updateClock();
  setInterval(updateClock, 1000);

  // Auto-refresh summary every 5 minutes (cache TTL-aligned)
  setInterval(loadSummary, 5 * 60 * 1000);

  bindEvents();

  // Load income first so computeFilteredSummary can exclude liquidations
  // on the very first render; trades table can load in parallel after.
  await loadIncome();
  await Promise.all([loadSummary(), loadTrades()]);
}

document.addEventListener('DOMContentLoaded', init);

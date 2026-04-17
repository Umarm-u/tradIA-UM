/**
 * tradIA Dashboard — Main Application
 * ====================================
 * Consumes the dashboard.py Flask API (read-only).
 * Does NOT call any trading logic or modify backend data.
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

function fmtPct(n, dec = 2) {
  const num = parseFloat(n);
  if (isNaN(num)) return '—';
  const s = (num >= 0 ? '+' : '') + num.toFixed(dec) + '%';
  return s;
}

function fmtPrice(n) {
  return '$' + fmt(n, 2);
}

function fmtDatetime(str) {
  if (!str) return '—';
  try {
    const d = new Date(str.replace ? str.replace(/ /, 'T').replace(/\+00:00$/, 'Z') : str);
    if (isNaN(d.getTime())) return str;
    return d.toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
  } catch { return str; }
}

function fmtDateShort(str) {
  if (!str) return '—';
  try {
    const d = new Date(str.replace ? str.replace(/ /, 'T').replace(/\+00:00$/, 'Z') : str);
    if (isNaN(d.getTime())) return str;
    return d.toISOString().slice(0, 10);
  } catch { return str; }
}

async function apiFetch(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`API ${path} returned ${r.status}`);
  return r.json();
}

/* ================================================================
   Toast Notifications
   ================================================================ */
function showToast(message, type = 'info', duration = 3500) {
  const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ'}</span>
                  <span class="toast-msg">${escHtml(message)}</span>`;
  document.getElementById('toast-container').appendChild(el);
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('visible')));
  setTimeout(() => {
    el.classList.remove('visible');
    setTimeout(() => el.remove(), 350);
  }, duration);
}

/* ================================================================
   Application State
   ================================================================ */
const S = {
  price: null,
  prevPrice: null,
  ticker24h: null,
  status: null,
  performance: null,
  trades: [],
  liveTrades: [],
  equity: [],
  logs: [],
  config: {},

  // Trade table — shared controls
  tradeSource: 'backtest',    // 'backtest' | 'live'
  tradeFilter: 'ALL',
  tradeSearch: '',

  // Backtest table state
  tradeSortKey: 'entry_time',
  tradeSortDir: 'desc',
  tradePage: 1,
  tradePageSize: 20,

  // Live trades table state
  liveSortKey: 'entry_time',
  liveSortDir: 'desc',
  livePage: 1,
  livePageSize: 20,

  // Equity chart
  equityTf: 'ALL',

  // Logs
  logsAutoScroll: true,
  logsLevelFilter: 'ALL',

  // Price chart
  chartInterval: '15m',
  chartCandles: [],

  // Active breakdown tab
  breakdownTab: 'BUY',

  // Connection stability — only flip offline after 3 consecutive failures
  connFailCount: 0,
};

/* ================================================================
   Price Chart (TradingView Lightweight Charts)
   ================================================================ */
let priceChart = null;
let candleSeries = null;
let volumeSeries = null;
let ema20Series  = null;
let ema50Series  = null;

function initPriceChart() {
  const { createChart, CrosshairMode, LineStyle } = LightweightCharts;
  const container = document.getElementById('chart-container');

  priceChart = createChart(container, {
    width:  container.clientWidth,
    height: container.clientHeight || 400,
    layout: {
      background: { color: 'transparent' },
      textColor:  '#64748b',
      fontSize:   11,
    },
    grid: {
      vertLines: { color: 'rgba(26,39,66,0.8)' },
      horzLines: { color: 'rgba(26,39,66,0.8)' },
    },
    crosshair: {
      mode: CrosshairMode.Normal,
      vertLine: { color: 'rgba(100,116,139,0.5)', style: LineStyle.Dashed },
      horzLine: { color: 'rgba(100,116,139,0.5)', style: LineStyle.Dashed },
    },
    rightPriceScale: {
      borderColor: '#1a2742',
      textColor:   '#64748b',
    },
    timeScale: {
      borderColor:     '#1a2742',
      textColor:       '#64748b',
      timeVisible:     true,
      secondsVisible:  false,
      fixLeftEdge:     true,
      fixRightEdge:    true,
    },
    handleScroll:   { vertTouchDrag: false },
    handleScale:    true,
  });

  candleSeries = priceChart.addCandlestickSeries({
    upColor:       '#0ecb81',
    downColor:     '#f6465d',
    borderVisible: false,
    wickUpColor:   '#0ecb81',
    wickDownColor: '#f6465d',
  });

  volumeSeries = priceChart.addHistogramSeries({
    priceFormat:     { type: 'volume' },
    priceScaleId:    'volume',
    scaleMargins:    { top: 0.85, bottom: 0 },
    lastValueVisible: false,
    priceLineVisible: false,
  });

  ema20Series = priceChart.addLineSeries({
    color:           'rgba(24,144,255,0.7)',
    lineWidth:       1,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });

  ema50Series = priceChart.addLineSeries({
    color:           'rgba(240,185,11,0.6)',
    lineWidth:       1,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });

  // OHLCV hover tooltip
  priceChart.subscribeCrosshairMove((param) => {
    const bar = param.seriesData?.get(candleSeries);
    if (bar) {
      $('#chart-o').textContent = `O: ${bar.open.toFixed(2)}`;
      $('#chart-h').textContent = `H: ${bar.high.toFixed(2)}`;
      $('#chart-l').textContent = `L: ${bar.low.toFixed(2)}`;
      $('#chart-c').textContent = `C: ${bar.close.toFixed(2)}`;
      const col = bar.close >= bar.open ? '#0ecb81' : '#f6465d';
      $('#chart-c').style.color = col;
    }
  });

  // Resize observer
  new ResizeObserver(() => {
    priceChart.applyOptions({
      width:  container.clientWidth,
      height: container.clientHeight,
    });
  }).observe(container);
}

function calcEMA(data, period) {
  const k = 2 / (period + 1);
  const result = [];
  let ema = null;
  for (const d of data) {
    if (ema === null) {
      ema = d.close;
    } else {
      ema = d.close * k + ema * (1 - k);
    }
    if (result.length >= period - 1) {
      result.push({ time: d.time, value: parseFloat(ema.toFixed(2)) });
    }
  }
  return result;
}

async function loadCandles() {
  try {
    const data = await apiFetch(`/api/candles?symbol=BTCUSDT&interval=${S.chartInterval}&limit=300`);
    if (!Array.isArray(data) || data.length === 0) return;

    S.chartCandles = data;

    const candles = data.map(d => ({
      time:  d.time,
      open:  d.open,
      high:  d.high,
      low:   d.low,
      close: d.close,
    }));

    const volumes = data.map(d => ({
      time:  d.time,
      value: d.volume,
      color: d.close >= d.open ? 'rgba(14,203,129,0.3)' : 'rgba(246,70,93,0.3)',
    }));

    candleSeries.setData(candles);
    volumeSeries.setData(volumes);
    ema20Series.setData(calcEMA(data, 20));
    ema50Series.setData(calcEMA(data, 50));

    priceChart.timeScale().fitContent();
    setConnection(true);
  } catch (e) {
    setConnection(false);
    console.warn('Candles fetch failed:', e);
  }
}

/* ================================================================
   Live Price
   ================================================================ */
async function loadPrice() {
  try {
    const [priceData, ticker] = await Promise.all([
      apiFetch('/api/price'),
      apiFetch('/api/ticker24h'),
    ]);

    S.prevPrice = S.price;
    S.price     = priceData.price;
    S.ticker24h = ticker;

    updatePriceTicker();
    S.connFailCount = 0;
    setConnection(true);

    // Update live price line on chart
    if (candleSeries && S.price) {
      candleSeries.applyOptions({
        lastValueVisible: true,
        priceLineVisible: true,
        priceLineColor:   'rgba(100,116,139,0.6)',
        priceLineWidth:   1,
        priceLineStyle:   0,
      });
    }

    // Update position unrealized PnL
    if (S.status?.position) renderPositionPnl();

  } catch (e) {
    S.connFailCount++;
    if (S.connFailCount >= 3) setConnection(false);
  }
}

function updatePriceTicker() {
  const el   = $('#ticker-price');
  const chEl = $('#ticker-change');
  if (!el) return;

  el.textContent = S.price ? `$${fmt(S.price, 2)}` : '—';

  // Flash direction
  if (S.prevPrice !== null && S.price !== null) {
    el.classList.remove('tick-up', 'tick-down');
    if (S.price > S.prevPrice) {
      el.classList.add('tick-up');
    } else if (S.price < S.prevPrice) {
      el.classList.add('tick-down');
    }
    setTimeout(() => el.classList.remove('tick-up', 'tick-down'), 800);
  }

  if (S.ticker24h && chEl) {
    const pct = S.ticker24h.priceChangePct;
    chEl.textContent = fmtPct(pct);
    chEl.className = `nav-ticker__change ${pct >= 0 ? 'positive' : 'negative'}`;
  }

  // 24h stats
  if (S.ticker24h) {
    const h = $('#stat-24h-high');
    const l = $('#stat-24h-low');
    const v = $('#stat-24h-vol');
    if (h) h.textContent = fmtPrice(S.ticker24h.highPrice);
    if (l) l.textContent = fmtPrice(S.ticker24h.lowPrice);
    if (v) v.textContent = fmt(S.ticker24h.quoteVolume / 1e6, 0) + 'M';
  }
}

/* ================================================================
   Bot Status & Position
   ================================================================ */
async function loadStatus() {
  try {
    S.status = await apiFetch('/api/status');
    renderBotStatus();
    renderPosition();
    renderDailySession();
  } catch (e) {
    console.warn('Status fetch failed:', e);
  }
}

function renderBotStatus() {
  const badge = $('#bot-status-badge');
  const txt   = $('#bot-status-text');
  if (!badge) return;
  const running = S.status?.bot_running;
  badge.className = `bot-status-badge ${running ? 'running' : 'stopped'}`;
  if (txt) txt.textContent = running ? 'BOT LIVE' : 'BOT OFFLINE';
}

function renderPosition() {
  const card = $('#position-card');
  if (!card) return;

  const pos = S.status?.position;
  if (!pos) {
    card.className = 'position-card empty';
    card.innerHTML = '<span>No active position</span>';
    return;
  }

  const isLong    = pos.side === 'LONG';
  const sideClass = isLong ? 'long' : 'short';
  const sideColor = isLong ? 'pos-long' : 'pos-short';
  const sideIcon  = isLong ? '▲' : '▼';

  const trailing  = S.status?.trailing_state;
  const trailBadge = trailing?.is_active
    ? '<span class="position-badge trail-badge">TRAILING</span>'
    : '';

  card.className = `position-card ${sideClass}`;
  card.innerHTML = `
    <div class="position-header">
      <div class="position-side ${sideColor}">${sideIcon} ${escHtml(pos.side)} ${trailBadge}</div>
      <span class="label-sm" id="pos-pnl-label">—</span>
    </div>
    <div class="position-grid">
      <div class="pos-field">
        <span class="pos-field__label">Entry</span>
        <span class="pos-field__value mono">${fmtPrice(pos.entry_price)}</span>
      </div>
      <div class="pos-field">
        <span class="pos-field__label">Quantity</span>
        <span class="pos-field__value mono">${pos.quantity ?? '—'}</span>
      </div>
      <div class="pos-field">
        <span class="pos-field__label">Stop Loss</span>
        <span class="pos-field__value mono red">${fmtPrice(pos.sl_price)}</span>
      </div>
      <div class="pos-field">
        <span class="pos-field__label">Take Profit</span>
        <span class="pos-field__value mono green">${fmtPrice(pos.tp_price)}</span>
      </div>
    </div>
  `;
  renderPositionPnl();
}

function renderPositionPnl() {
  const el  = $('#pos-pnl-label');
  const pos = S.status?.position;
  if (!el || !pos || !S.price) return;

  const entry = parseFloat(pos.entry_price);
  const price = S.price;
  const pnlPct = pos.side === 'LONG'
    ? (price - entry) / entry * 100
    : (entry - price) / entry * 100;

  el.textContent = fmtPct(pnlPct);
  el.className   = `label-sm ${pnlPct >= 0 ? 'green' : 'red'}`;
}

function renderDailySession() {
  const dailyPnlEl = $('#daily-session-pnl');
  const dailyTxEl  = $('#daily-session-trades');
  if (!dailyPnlEl) return;

  const pnl    = S.status?.daily_pnl ?? 0;
  const trades = S.status?.daily_trades ?? 0;
  const cls    = pnl >= 0 ? 'green' : 'red';
  dailyPnlEl.innerHTML = `<span class="${cls}">${pnl >= 0 ? '+' : ''}${fmt(pnl, 4)} USDT</span>`;
  if (dailyTxEl) dailyTxEl.textContent = trades;
}

/* ================================================================
   Performance Metrics
   ================================================================ */
async function loadPerformance() {
  try {
    S.performance = await apiFetch('/api/performance');
    renderPerformance();
    renderRiskIndicators();
  } catch (e) {
    console.warn('Performance fetch failed:', e);
  }
}

function renderPerformance() {
  const p = S.performance;
  if (!p) return;

  const buy  = p.buy  || {};
  const sell = p.sell || {};
  const comb = p.combined || {};

  // Summary stats (combined-ish)
  const winRate     = comb.win_rate       ?? 0;
  const totalReturn = comb.total_return_pct ?? 0;
  const profitFactor = avg([buy.profit_factor, sell.profit_factor]);
  const sharpe       = avg([buy.sharpe_ratio, sell.sharpe_ratio]);
  const maxDD        = Math.min(
    parseFloat(buy.max_drawdown_pct ?? 0),
    parseFloat(sell.max_drawdown_pct ?? 0)
  );

  setStatCard('stat-winrate',     `${(winRate * 100).toFixed(1)}%`);
  setStatCard('stat-totalreturn', `+${totalReturn.toFixed(1)}%`, 'green');
  setStatCard('stat-pf',          profitFactor ? profitFactor.toFixed(2) : '—');
  setStatCard('stat-sharpe',      sharpe ? sharpe.toFixed(2) : '—', 'highlight');
  setStatCard('stat-maxdd',       `${maxDD.toFixed(2)}%`, 'red');
  setStatCard('stat-trades',      (comb.total_trades ?? 0).toString());

  // Breakdown tabs
  renderBreakdown();
}

function avg(arr) {
  const nums = arr.filter(v => v != null && !isNaN(parseFloat(v))).map(Number);
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
}

function setStatCard(id, value, cls) {
  const el = $(`#${id}`);
  if (!el) return;
  el.textContent = value;
  if (cls) {
    const card = el.closest('.stat-card');
    if (card) card.className = `stat-card ${cls}`;
  }
}

function renderBreakdown() {
  const tab = S.breakdownTab === 'BUY' ? 'buy' : 'sell';
  const r   = S.performance?.[tab] || {};

  const fields = [
    ['total_trades',    'Total Trades',    (v) => v],
    ['win_rate',        'Win Rate',        (v) => `${(parseFloat(v)*100).toFixed(1)}%`],
    ['avg_return_pct',  'Avg Return',      (v) => fmtPct(v)],
    ['avg_win_pct',     'Avg Win',         (v) => `+${parseFloat(v).toFixed(2)}%`],
    ['avg_loss_pct',    'Avg Loss',        (v) => `${parseFloat(v).toFixed(2)}%`],
    ['profit_factor',   'Profit Factor',   (v) => parseFloat(v).toFixed(2)],
    ['sharpe_ratio',    'Sharpe Ratio',    (v) => parseFloat(v).toFixed(2)],
    ['calmar_ratio',    'Calmar Ratio',    (v) => parseFloat(v).toFixed(2)],
    ['max_drawdown_pct','Max Drawdown',    (v) => `${parseFloat(v).toFixed(2)}%`],
    ['total_return_pct','Total Return',    (v) => `+${parseFloat(v).toFixed(2)}%`],
    ['avg_holding_bars','Avg Hold (bars)', (v) => parseFloat(v).toFixed(1)],
  ];

  const tbody = $('#breakdown-table-body');
  if (!tbody) return;

  tbody.innerHTML = fields.map(([key, label, fmt]) => {
    const raw = r[key];
    const val = raw != null ? fmt(raw) : '—';
    let valClass = '';
    if (key === 'win_rate' || key === 'avg_win_pct' || key === 'total_return_pct') valClass = 'green';
    if (key === 'avg_loss_pct' || key === 'max_drawdown_pct') valClass = 'red';
    if (key === 'profit_factor' || key === 'sharpe_ratio') valClass = 'accent';
    return `<tr>
      <td class="bd-row__key">${escHtml(label)}</td>
      <td class="bd-row__val ${valClass}">${escHtml(val)}</td>
    </tr>`;
  }).join('');

  // Exit reason bars
  renderExitBars(tab);
}

function renderExitBars(tab) {
  const r = S.performance?.[tab] || {};
  const reasons = r.exit_reasons;
  if (!reasons || typeof reasons !== 'string') return;

  // Parse string like "{'trail_SL': 48, 'SL': 40, 'TP': 2}"
  try {
    const clean = reasons.replace(/'/g, '"').replace(/(\w+):/g, '"$1":');
    const obj   = JSON.parse(clean);
    const total = Object.values(obj).reduce((a, b) => a + b, 0);

    const fill = (id, key, cls) => {
      const el = $(`#${id}`);
      if (!el) return;
      const cnt = obj[key] ?? 0;
      el.style.width = total ? `${(cnt / total * 100).toFixed(1)}%` : '0%';
      el.className = `er-bar-fill ${cls}`;
      const counter = $(`#${id}-count`);
      if (counter) counter.textContent = cnt;
    };

    fill('er-trail', 'trail_SL', 'er-fill-trail');
    fill('er-sl',    'SL',       'er-fill-sl');
    fill('er-tp',    'TP',       'er-fill-tp');
  } catch { /* ignore */ }
}

function renderRiskIndicators() {
  const buy  = S.performance?.buy  || {};
  const sell = S.performance?.sell || {};
  const maxDD = Math.abs(Math.min(
    parseFloat(buy.max_drawdown_pct ?? 0),
    parseFloat(sell.max_drawdown_pct ?? 0)
  ));

  const cfg     = S.config;
  const maxLoss = parseFloat(cfg.max_daily_loss ?? 0.05) * 100;
  const dailyPnl = S.status?.daily_pnl ?? 0;
  const dailyLossPct = Math.max(0, -dailyPnl);  // USDT loss today

  // Risk bar (daily loss as % of max allowed)
  const riskPct = maxLoss > 0 ? Math.min(dailyLossPct / maxLoss * 100, 100) : 0;
  const riskBar = $('#risk-daily-fill');
  const riskLbl = $('#risk-daily-pct');
  if (riskBar) {
    riskBar.style.width  = `${riskPct}%`;
    riskBar.className    = `risk-bar-fill ${riskPct < 50 ? 'safe' : riskPct < 80 ? 'warn' : 'danger'}`;
  }
  if (riskLbl) riskLbl.textContent = `${riskPct.toFixed(1)}%`;

  // Drawdown bar
  const ddMax = 10; // cap visual at 10% for display
  const ddBar = $('#risk-dd-fill');
  const ddLbl = $('#risk-dd-pct');
  if (ddBar) {
    const ddPct = Math.min(maxDD / ddMax * 100, 100);
    ddBar.style.width = `${ddPct}%`;
    ddBar.className   = `risk-bar-fill ${ddPct < 40 ? 'safe' : ddPct < 70 ? 'warn' : 'danger'}`;
  }
  if (ddLbl) ddLbl.textContent = `${maxDD.toFixed(2)}%`;

  // Alert banners
  const lossAlert = $('#alert-loss');
  const ddAlert   = $('#alert-dd');
  if (lossAlert) {
    lossAlert.classList.toggle('visible', riskPct >= 50);
  }
  // Drawdown alert: fire only when live session has consumed ≥ 75% of the
  // max-daily-loss budget — not based on static backtest drawdown figures.
  if (ddAlert) {
    ddAlert.classList.toggle('visible', riskPct >= 75);
  }
}

/* ================================================================
   Equity Curve (Chart.js)
   ================================================================ */
let equityChart = null;

function initEquityChart() {
  const ctx = document.getElementById('equity-canvas')?.getContext('2d');
  if (!ctx) return;

  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Equity',
        data: [],
        borderColor:     '#1890ff',
        backgroundColor: (c) => {
          const g = c.chart.ctx.createLinearGradient(0, 0, 0, 260);
          g.addColorStop(0,   'rgba(24,144,255,0.20)');
          g.addColorStop(1,   'rgba(24,144,255,0.00)');
          return g;
        },
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: '#1890ff',
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0d1421',
          borderColor:     '#1a2742',
          borderWidth:     1,
          titleColor:      '#94a3b8',
          bodyColor:       '#e2e8f0',
          padding:         10,
          callbacks: {
            title:  (items) => fmtDateShort(items[0]?.label),
            label:  (item)  => `Equity: ${item.raw?.y?.toFixed(2) ?? '—'}`,
          },
        },
      },
      scales: {
        x: {
          type:  'time',
          time:  { unit: 'month', tooltipFormat: 'yyyy-MM-dd' },
          grid:  { color: 'rgba(26,39,66,0.8)' },
          ticks: { color: '#4a5568', maxTicksLimit: 8, maxRotation: 0 },
        },
        y: {
          position: 'right',
          grid:     { color: 'rgba(26,39,66,0.8)' },
          ticks:    {
            color:    '#4a5568',
            callback: (v) => v.toFixed(0),
          },
        },
      },
    },
  });
}

async function loadEquity() {
  try {
    S.equity = await apiFetch('/api/equity');
    renderEquityChart();
  } catch (e) {
    console.warn('Equity fetch failed:', e);
  }
}

function renderEquityChart() {
  if (!equityChart || !S.equity.length) return;

  const now    = new Date(S.equity[S.equity.length - 1]?.time || new Date());
  const cutoff = getEquityCutoff(S.equityTf, now);

  const filtered = S.equity
    .filter(p => {
      if (!cutoff) return true;
      return new Date(p.time.replace(' ', 'T').replace(/\+00:00$/, 'Z')) >= cutoff;
    })
    .map(p => ({
      x: new Date(p.time.replace(' ', 'T').replace(/\+00:00$/, 'Z')),
      y: p.value,
    }));

  equityChart.data.datasets[0].data = filtered;
  equityChart.update('none');

  // Update equity summary stats
  if (filtered.length >= 2) {
    const start  = filtered[0].y;
    const end    = filtered[filtered.length - 1].y;
    const retPct = ((end - start) / start * 100).toFixed(2);
    const cls    = retPct >= 0 ? 'green' : 'red';
    const el     = $('#equity-return-label');
    if (el) el.innerHTML = `<span class="${cls}">${retPct >= 0 ? '+' : ''}${retPct}%</span>`;
    const endEl = $('#equity-final');
    if (endEl) endEl.textContent = end.toFixed(2);
  }
}

function getEquityCutoff(tf, now) {
  const d = new Date(now);
  if (tf === 'ALL') return null;
  if (tf === '3M')  { d.setMonth(d.getMonth() - 3); return d; }
  if (tf === '6M')  { d.setMonth(d.getMonth() - 6); return d; }
  if (tf === '1Y')  { d.setFullYear(d.getFullYear() - 1); return d; }
  if (tf === '2Y')  { d.setFullYear(d.getFullYear() - 2); return d; }
  return null;
}

/* ================================================================
   Trade History Table
   ================================================================ */
async function loadTrades() {
  try {
    S.trades = await apiFetch('/api/trades');
    renderTradesTable();
    const badge = $('#bt-trade-count');
    if (badge) badge.textContent = S.trades.length;
  } catch (e) {
    console.warn('Trades fetch failed:', e);
  }
}

function getFilteredTrades() {
  let t = [...S.trades];

  if (S.tradeFilter !== 'ALL') {
    t = t.filter(r => r.direction === S.tradeFilter);
  }

  if (S.tradeSearch) {
    const q = S.tradeSearch.toLowerCase();
    t = t.filter(r =>
      (r.direction || '').toLowerCase().includes(q) ||
      (r.exit_reason || '').toLowerCase().includes(q) ||
      (r.entry_time  || '').toLowerCase().includes(q)
    );
  }

  // Sort
  t.sort((a, b) => {
    let av = a[S.tradeSortKey] ?? '';
    let bv = b[S.tradeSortKey] ?? '';
    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    if (av < bv) return S.tradeSortDir === 'asc' ? -1 :  1;
    if (av > bv) return S.tradeSortDir === 'asc' ?  1 : -1;
    return 0;
  });

  return t;
}

function renderTradesTable() {
  const filtered = getFilteredTrades();
  const total    = filtered.length;
  const pages    = Math.max(1, Math.ceil(total / S.tradePageSize));
  S.tradePage    = Math.min(S.tradePage, pages);

  const start = (S.tradePage - 1) * S.tradePageSize;
  const slice = filtered.slice(start, start + S.tradePageSize);

  const countEl = $('#trades-count');
  if (countEl) countEl.textContent = `${total} trades`;

  const tbody = $('#trades-tbody');
  if (!tbody) return;

  if (slice.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:30px;color:var(--text-3);">No trades match your filter</td></tr>`;
  } else {
    tbody.innerHTML = slice.map(t => {
      const pnlCls = t.net_pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative';
      const pnlStr = (t.net_pnl_pct >= 0 ? '+' : '') + t.net_pnl_pct.toFixed(3) + '%';
      const dirCls = t.direction === 'BUY' ? 'buy' : 'sell';
      const dirIcon = t.direction === 'BUY' ? '▲' : '▼';
      const exitCls = (t.exit_reason || '').toLowerCase().replace('_', '_');
      const confPct = (t.confidence * 100).toFixed(1) + '%';
      const holdH   = (t.holding_bars * 0.25).toFixed(1) + 'h';
      return `<tr>
        <td>${escHtml(fmtDateShort(t.entry_time))}</td>
        <td><span class="dir-badge ${dirCls}">${dirIcon} ${escHtml(t.direction)}</span></td>
        <td>${fmtPrice(t.entry_price)}</td>
        <td>${fmtPrice(t.exit_price)}</td>
        <td class="${pnlCls}" style="font-weight:600">${escHtml(pnlStr)}</td>
        <td><span class="exit-badge ${exitCls}">${escHtml(t.exit_reason)}</span></td>
        <td>${escHtml(confPct)}</td>
        <td>${escHtml(holdH)}</td>
        <td>${t.trail_activated ? '<span class="badge badge-blue">✓</span>' : '<span style="color:var(--text-3)">—</span>'}</td>
      </tr>`;
    }).join('');
  }

  renderPagination(pages);

  // Update sort arrows
  $$('.data-table th[data-sort]').forEach(th => {
    const arrow = th.querySelector('.sort-arrow');
    const isActive = th.dataset.sort === S.tradeSortKey;
    th.classList.toggle('sorted', isActive);
    if (arrow) arrow.textContent = isActive
      ? (S.tradeSortDir === 'asc' ? '↑' : '↓')
      : '↕';
  });
}

function renderPagination(pages) {
  const pg = $('#trades-pagination');
  if (!pg) return;

  let html = `<button class="page-btn" onclick="changePage(${S.tradePage - 1})" ${S.tradePage <= 1 ? 'disabled' : ''}>‹</button>`;

  const range = getPaginationRange(S.tradePage, pages);
  for (const p of range) {
    if (p === '…') {
      html += `<span style="padding:0 6px;color:var(--text-3)">…</span>`;
    } else {
      html += `<button class="page-btn ${p === S.tradePage ? 'active' : ''}" onclick="changePage(${p})">${p}</button>`;
    }
  }

  html += `<button class="page-btn" onclick="changePage(${S.tradePage + 1})" ${S.tradePage >= pages ? 'disabled' : ''}>›</button>`;
  pg.innerHTML = html;
}

function getPaginationRange(current, total) {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const range = [1];
  if (current > 3) range.push('…');
  for (let i = Math.max(2, current - 1); i <= Math.min(total - 1, current + 1); i++) range.push(i);
  if (current < total - 2) range.push('…');
  range.push(total);
  return range;
}

window.changePage = (p) => {
  const pages = Math.ceil(getFilteredTrades().length / S.tradePageSize);
  if (p < 1 || p > pages) return;
  S.tradePage = p;
  renderTradesTable();
};

/* ================================================================
   Activity Logs
   ================================================================ */
async function loadLogs() {
  try {
    S.logs = await apiFetch('/api/logs');
    renderLogs();
  } catch (e) {
    console.warn('Logs fetch failed:', e);
  }
}

function renderLogs() {
  const console_ = document.getElementById('logs-console');
  if (!console_) return;

  let logs = S.logs;

  if (S.logsLevelFilter !== 'ALL') {
    logs = logs.filter(l => (l.level || '').toUpperCase() === S.logsLevelFilter);
  }

  if (logs.length === 0) {
    console_.innerHTML = `
      <div class="logs-empty">
        <div>
          <div style="font-size:24px;margin-bottom:8px;opacity:.3">📋</div>
          <div>No logs available</div>
          <div style="font-size:10px;margin-top:4px;color:var(--text-3)">Start the bot to see activity here</div>
        </div>
      </div>`;
    return;
  }

  const wasAtBottom = console_.scrollHeight - console_.scrollTop < console_.clientHeight + 40;

  console_.innerHTML = logs.map(l => {
    const level  = (l.level || 'INFO').toUpperCase();
    const lvlCls = level === 'WARNING' ? 'warning' : level === 'ERROR' ? 'error' : level === 'DEBUG' ? 'debug' : 'info';
    const msg    = escHtml(l.message || '');
    const ts     = l.timestamp ? l.timestamp.slice(11, 19) : '';
    return `<div class="log-entry ${lvlCls}">
      <span class="log-ts">${escHtml(ts)}</span>
      <span class="log-level">${escHtml(level)}</span>
      <span class="log-msg">${msg}</span>
    </div>`;
  }).join('');

  if (S.logsAutoScroll && wasAtBottom) {
    console_.scrollTop = console_.scrollHeight;
  }
}

/* ================================================================
   Controls Panel
   ================================================================ */
async function loadConfig() {
  try {
    S.config = await apiFetch('/api/config');
    renderControls();
  } catch (e) {
    console.warn('Config fetch failed:', e);
  }
}

function renderControls() {
  const c = S.config;
  if (!c) return;

  const riskSlider = $('#ctrl-risk');
  const riskVal    = $('#ctrl-risk-val');
  if (riskSlider) {
    riskSlider.value   = c.risk_per_trade * 100;
    if (riskVal) riskVal.textContent = `${(c.risk_per_trade * 100).toFixed(1)}%`;
  }

  const lossSlider = $('#ctrl-loss');
  const lossVal    = $('#ctrl-loss-val');
  if (lossSlider) {
    lossSlider.value   = c.max_daily_loss * 100;
    if (lossVal) lossVal.textContent = `${(c.max_daily_loss * 100).toFixed(1)}%`;
  }

  const buySlider = $('#ctrl-buy-thresh');
  const buyVal    = $('#ctrl-buy-thresh-val');
  if (buySlider) {
    buySlider.value = c.buy_threshold * 100;
    if (buyVal) buyVal.textContent = c.buy_threshold.toFixed(3);
  }

  const sellSlider = $('#ctrl-sell-thresh');
  const sellVal    = $('#ctrl-sell-thresh-val');
  if (sellSlider) {
    sellSlider.value = c.sell_threshold * 100;
    if (sellVal) sellVal.textContent = c.sell_threshold.toFixed(3);
  }

  // Mode toggle
  const dryBtn  = $('#mode-dry');
  const liveBtn = $('#mode-live');
  if (dryBtn && liveBtn) {
    const isDry = c.dry_run;
    dryBtn.classList.toggle('active', isDry);
    liveBtn.classList.toggle('active', !isDry);
    S._pendingDryRun = isDry;
  }
}

function getCurrentControlValues() {
  return {
    risk_per_trade: parseFloat($('#ctrl-risk')?.value ?? 2) / 100,
    max_daily_loss: parseFloat($('#ctrl-loss')?.value ?? 5) / 100,
    buy_threshold:  parseFloat($('#ctrl-buy-thresh')?.value ?? 60) / 100,
    sell_threshold: parseFloat($('#ctrl-sell-thresh')?.value ?? 77.5) / 100,
    dry_run:        S._pendingDryRun !== undefined ? S._pendingDryRun : (S.config?.dry_run ?? true),
  };
}

function formatChanges(pending, current) {
  const labels = {
    risk_per_trade: 'Risk / Trade',
    max_daily_loss: 'Max Daily Loss',
    buy_threshold:  'BUY Threshold',
    sell_threshold: 'SELL Threshold',
    dry_run:        'Mode',
  };
  const fmt = (k, v) => {
    if (k === 'dry_run') return v ? 'DRY RUN' : '⚠ LIVE TRADING';
    if (k.includes('threshold')) return v.toFixed(3);
    return `${(v * 100).toFixed(1)}%`;
  };

  return Object.entries(pending)
    .filter(([k, v]) => current[k] !== undefined && String(v) !== String(current[k]))
    .map(([k, v]) => `${labels[k] ?? k}: ${fmt(k, current[k])} → ${fmt(k, v)}`)
    .join('\n') || 'No changes detected';
}

/* ================================================================
   Confirmation Modal
   ================================================================ */
let _modalResolve = null;

function showModal({ title, body, bodyHtml = null, confirmText = 'Confirm', danger = false }) {
  return new Promise(resolve => {
    _modalResolve = resolve;
    $('#modal-title').textContent   = title;
    if (bodyHtml) {
      $('#modal-body').innerHTML  = bodyHtml;
    } else {
      $('#modal-body').textContent = body;
    }
    $('#modal-confirm').textContent = confirmText;
    $('#modal-confirm').className   = `btn-confirm${danger ? ' danger' : ''}`;
    $('#modal-overlay').classList.add('visible');
  });
}

function closeModal(result) {
  $('#modal-overlay').classList.remove('visible');
  if (_modalResolve) { _modalResolve(result); _modalResolve = null; }
}

/* ================================================================
   Connection indicator
   ================================================================ */
function setConnection(online) {
  const dot  = $('#conn-dot');
  const text = $('#conn-text');
  if (!dot) return;
  dot.className = `conn-dot ${online ? 'online' : 'offline'}`;
  if (text) text.textContent = online ? 'Live' : 'Offline';
}

/* ================================================================
   Clock
   ================================================================ */
function updateClock() {
  const el = $('#nav-clock');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toUTCString().slice(17, 25) + ' UTC';
}

/* ================================================================
   Live Trades Table
   ================================================================ */
async function loadLiveTrades() {
  try {
    S.liveTrades = await apiFetch('/api/live_trades');
    renderLiveTradesTable();
    // Update badge count
    const badge = $('#live-trade-count');
    if (badge) badge.textContent = S.liveTrades.length;
  } catch (e) {
    console.warn('Live trades fetch failed:', e);
  }
}

function getLiveFiltered() {
  let t = [...S.liveTrades];
  if (S.tradeFilter !== 'ALL') {
    t = t.filter(r => r.direction === S.tradeFilter);
  }
  if (S.tradeSearch) {
    const q = S.tradeSearch.toLowerCase();
    t = t.filter(r =>
      (r.direction   || '').toLowerCase().includes(q) ||
      (r.entry_time  || '').toLowerCase().includes(q) ||
      (r.exit_time   || '').toLowerCase().includes(q)
    );
  }
  t.sort((a, b) => {
    let av = a[S.liveSortKey] ?? '';
    let bv = b[S.liveSortKey] ?? '';
    if (typeof av === 'string') av = av.toLowerCase();
    if (typeof bv === 'string') bv = bv.toLowerCase();
    if (av < bv) return S.liveSortDir === 'asc' ? -1 :  1;
    if (av > bv) return S.liveSortDir === 'asc' ?  1 : -1;
    return 0;
  });
  return t;
}

function renderLiveTradesTable() {
  const filtered = getLiveFiltered();
  const total    = filtered.length;
  const pages    = Math.max(1, Math.ceil(total / S.livePageSize));
  S.livePage     = Math.min(S.livePage, pages);

  const start = (S.livePage - 1) * S.livePageSize;
  const slice = filtered.slice(start, start + S.livePageSize);

  const countEl = $('#trades-count');
  if (countEl) countEl.textContent = `${total} live trades`;

  const tbody = $('#live-trades-tbody');
  if (!tbody) return;

  if (slice.length === 0) {
    const msg = S.liveTrades.length === 0
      ? 'No live trades found — run <code>python run_live_bot.py</code> to start trading'
      : 'No trades match your filter';
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:40px;color:var(--text-3);line-height:2">${msg}</td></tr>`;
    $('#live-trades-pagination').innerHTML = '';
    return;
  }

  tbody.innerHTML = slice.map(t => {
    const dirCls   = (t.direction || '').toLowerCase() === 'long' || t.direction === 'BUY' ? 'buy' : 'sell';
    const dirLabel = t.direction === 'LONG' ? '▲ LONG' : t.direction === 'SHORT' ? '▼ SHORT' : t.direction;
    const pnl      = t.pnl_usdt ?? null;
    const pnlCls   = pnl === null ? '' : pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
    const pnlStr   = pnl === null ? '—' : `${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)} USDT`;
    const conf     = t.confidence != null ? (t.confidence * 100).toFixed(1) + '%' : '—';
    const trail    = t.trail_activated
      ? '<span class="badge badge-blue">✓</span>'
      : '<span style="color:var(--text-3)">—</span>';

    return `<tr>
      <td style="font-family:var(--font-ui);font-size:11px">${escHtml(fmtDateShort(t.entry_time))}</td>
      <td><span class="dir-badge ${dirCls}">${escHtml(dirLabel)}</span></td>
      <td>${t.entry_price ? fmtPrice(t.entry_price) : '—'}</td>
      <td class="red">${t.sl_price ? fmtPrice(t.sl_price) : '—'}</td>
      <td class="green">${t.tp_price ? fmtPrice(t.tp_price) : '—'}</td>
      <td class="${pnlCls}" style="font-weight:600">${escHtml(pnlStr)}</td>
      <td style="font-family:var(--font-ui);font-size:11px">${escHtml(fmtDateShort(t.exit_time))}</td>
      <td>${escHtml(conf)}</td>
      <td>${trail}</td>
    </tr>`;
  }).join('');

  // Pagination
  const pg = $('#live-trades-pagination');
  if (pg) {
    if (pages <= 1) { pg.innerHTML = ''; return; }
    let html = `<button class="page-btn" onclick="changeLivePage(${S.livePage - 1})" ${S.livePage <= 1 ? 'disabled' : ''}>‹</button>`;
    for (const p of getPaginationRange(S.livePage, pages)) {
      if (p === '…') html += `<span style="padding:0 6px;color:var(--text-3)">…</span>`;
      else html += `<button class="page-btn ${p === S.livePage ? 'active' : ''}" onclick="changeLivePage(${p})">${p}</button>`;
    }
    html += `<button class="page-btn" onclick="changeLivePage(${S.livePage + 1})" ${S.livePage >= pages ? 'disabled' : ''}>›</button>`;
    pg.innerHTML = html;
  }

  // Sort arrows
  $$('.data-table th[data-sort-live]').forEach(th => {
    const arrow = th.querySelector('.sort-arrow');
    const isActive = th.dataset.sortLive === S.liveSortKey;
    th.classList.toggle('sorted', isActive);
    if (arrow) arrow.textContent = isActive
      ? (S.liveSortDir === 'asc' ? '↑' : '↓')
      : '↕';
  });
}

window.changeLivePage = (p) => {
  const pages = Math.ceil(getLiveFiltered().length / S.livePageSize);
  if (p < 1 || p > pages) return;
  S.livePage = p;
  renderLiveTradesTable();
};

function switchTradeSource(source) {
  S.tradeSource = source;
  S.tradePage   = 1;
  S.livePage    = 1;

  $$('[data-source]').forEach(t => t.classList.toggle('active', t.dataset.source === source));

  const btWrap   = $('#backtest-table-wrap');
  const liveWrap = $('#live-table-wrap');
  if (btWrap)   btWrap.style.display   = source === 'backtest' ? '' : 'none';
  if (liveWrap) liveWrap.style.display = source === 'live'     ? '' : 'none';

  if (source === 'backtest') {
    renderTradesTable();
  } else {
    renderLiveTradesTable();
  }
}

/* ================================================================
   Event Bindings
   ================================================================ */
function bindAll() {
  // Interval buttons
  $$('.interval-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      S.chartInterval = btn.dataset.interval;
      $$('.interval-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      await loadCandles();
    });
  });

  // Equity time-filter buttons
  $$('.tf-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      S.equityTf = btn.dataset.tf;
      $$('.tf-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderEquityChart();
    });
  });

  // Trade direction filter — works for both tables
  $$('.dir-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      S.tradeFilter = btn.dataset.dir;
      S.tradePage   = 1;
      S.livePage    = 1;
      $$('.dir-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (S.tradeSource === 'backtest') renderTradesTable();
      else renderLiveTradesTable();
    });
  });

  // Trade search — works for both tables
  const search = $('#trade-search');
  if (search) {
    search.addEventListener('input', () => {
      S.tradeSearch = search.value.trim();
      S.tradePage   = 1;
      S.livePage    = 1;
      if (S.tradeSource === 'backtest') renderTradesTable();
      else renderLiveTradesTable();
    });
  }

  // Table column sort
  $$('.data-table th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (S.tradeSortKey === key) {
        S.tradeSortDir = S.tradeSortDir === 'asc' ? 'desc' : 'asc';
      } else {
        S.tradeSortKey = key;
        S.tradeSortDir = 'desc';
      }
      S.tradePage = 1;
      renderTradesTable();
    });
  });

  // Log level filter
  $$('.lvl-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      S.logsLevelFilter = btn.dataset.level;
      $$('.lvl-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderLogs();
    });
  });

  // Auto-scroll toggle
  const asTgl = $('#autoscroll-toggle');
  if (asTgl) {
    asTgl.addEventListener('click', () => {
      S.logsAutoScroll = !S.logsAutoScroll;
      const sw = asTgl.querySelector('.toggle-switch');
      if (sw) sw.classList.toggle('on', S.logsAutoScroll);
    });
  }

  // Trade source tabs (BACKTEST / LIVE)
  $$('[data-source]').forEach(tab => {
    tab.addEventListener('click', () => switchTradeSource(tab.dataset.source));
  });

  // Live trades column sort
  $$('.data-table th[data-sort-live]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sortLive;
      if (S.liveSortKey === key) {
        S.liveSortDir = S.liveSortDir === 'asc' ? 'desc' : 'asc';
      } else {
        S.liveSortKey = key;
        S.liveSortDir = 'desc';
      }
      S.livePage = 1;
      renderLiveTradesTable();
    });
  });

  // Breakdown tabs
  $$('.bdtab').forEach(tab => {
    if (tab.dataset.tab) {  // only analytics breakdown tabs have data-tab
      tab.addEventListener('click', () => {
        S.breakdownTab = tab.dataset.tab;
        $$('.bdtab[data-tab]').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        renderBreakdown();
      });
    }
  });

  // Control sliders live update
  const ctrlDefs = [
    ['#ctrl-risk',       '#ctrl-risk-val',        (v) => `${parseFloat(v).toFixed(1)}%`],
    ['#ctrl-loss',       '#ctrl-loss-val',         (v) => `${parseFloat(v).toFixed(1)}%`],
    ['#ctrl-buy-thresh', '#ctrl-buy-thresh-val',   (v) => (parseFloat(v) / 100).toFixed(3)],
    ['#ctrl-sell-thresh','#ctrl-sell-thresh-val',  (v) => (parseFloat(v) / 100).toFixed(3)],
  ];
  ctrlDefs.forEach(([slid, disp, fmt]) => {
    const el = $(slid);
    if (!el) return;
    el.addEventListener('input', () => {
      const d = $(disp);
      if (d) d.textContent = fmt(el.value);
    });
  });

  // Mode toggle
  $('#mode-dry')?.addEventListener('click',  () => setMode(true));
  $('#mode-live')?.addEventListener('click', () => setMode(false));

  // Save button
  $('#save-config-btn')?.addEventListener('click', handleSaveConfig);

  // Modal buttons
  $('#modal-confirm')?.addEventListener('click', () => closeModal(true));
  $('#modal-cancel')?.addEventListener('click',  () => closeModal(false));
  $('#modal-overlay')?.addEventListener('click', (e) => {
    if (e.target === $('#modal-overlay')) closeModal(false);
  });
}

function setMode(isDry) {
  S._pendingDryRun = isDry;
  $('#mode-dry')?.classList.toggle('active', isDry);
  $('#mode-live')?.classList.toggle('active', !isDry);
  if (!isDry) {
    showToast('⚠ LIVE mode selected — save to apply', 'warning');
  }
}

async function handleSaveConfig() {
  const pending = getCurrentControlValues();

  // Build diff summary
  const currentCopy = {
    risk_per_trade: S.config.risk_per_trade,
    max_daily_loss: S.config.max_daily_loss,
    buy_threshold:  S.config.buy_threshold,
    sell_threshold: S.config.sell_threshold,
    dry_run:        S.config.dry_run,
  };

  const changesText = formatChanges(pending, currentCopy);
  if (changesText === 'No changes detected') {
    showToast('No changes to save', 'info');
    return;
  }

  const isLiveSwitch = !pending.dry_run && S.config.dry_run;

  const confirmed = await showModal({
    title:       isLiveSwitch ? '⚠ Enable Live Trading' : 'Confirm Configuration Change',
    body:        isLiveSwitch
      ? 'You are about to switch to LIVE trading mode. Real funds will be used. Are you sure?'
      : undefined,
    bodyHtml:    !isLiveSwitch
      ? `The following changes will be written to .env:<div class="modal-changes">${escHtml(changesText)}</div>`
      : undefined,
    confirmText: isLiveSwitch ? 'Enable Live' : 'Save Changes',
    danger:      isLiveSwitch,
  });

  if (!confirmed) return;

  const btn = $('#save-config-btn');
  if (btn) btn.disabled = true;

  try {
    const r = await fetch('/api/config', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(pending),
    });
    const data = await r.json();
    if (data.success) {
      Object.assign(S.config, pending);
      showToast('Configuration saved successfully', 'success');
    } else {
      showToast(`Save failed: ${data.error || 'Unknown error'}`, 'error');
    }
  } catch (e) {
    showToast(`Network error: ${e.message}`, 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ================================================================
   Initialisation
   ================================================================ */
async function init() {
  // Init charts
  initPriceChart();
  initEquityChart();

  // Bind all interactive elements
  bindAll();

  // Kick off autoscroll toggle to on state
  const sw = $('#autoscroll-toggle')?.querySelector('.toggle-switch');
  if (sw) sw.classList.add('on');

  // Clock
  updateClock();
  setInterval(updateClock, 1000);

  // Show skeleton loading state
  const statsCards = $$('.stat-card__value');
  statsCards.forEach(el => el.classList.add('skeleton', 'skel-stat'));

  // Load all data in parallel
  await Promise.allSettled([
    loadPrice(),
    loadCandles(),
    loadStatus(),
    loadPerformance(),
    loadTrades(),
    loadLiveTrades(),
    loadLogs(),
    loadConfig(),
    loadEquity(),
  ]);

  // Remove skeletons
  statsCards.forEach(el => el.classList.remove('skeleton', 'skel-stat'));

  // Set up polling intervals
  setInterval(loadPrice,       3000);   // price: 3s
  setInterval(loadCandles,    60000);   // candles: 1m
  setInterval(loadStatus,     10000);   // bot status: 10s
  setInterval(loadLogs,       20000);   // logs: 20s
  setInterval(loadLiveTrades, 30000);   // live trades: 30s
  setInterval(loadPerformance, 300000); // perf: 5m (static data)
}

document.addEventListener('DOMContentLoaded', init);

import { useEffect, useState } from 'react';

// Primary / secondary team colors, used for the probability bar
export const TEAM_COLORS = {
  ATL: ['#E03A3E', '#26282A'], BOS: ['#007A33', '#BA9653'], BKN: ['#1D1D1D', '#707271'],
  CHA: ['#1D1160', '#00788C'], CHI: ['#CE1141', '#1D1D1D'], CLE: ['#860038', '#FDBB30'],
  DAL: ['#00538C', '#B8C4CA'], DEN: ['#0E2240', '#FEC524'], DET: ['#C8102E', '#1D42BA'],
  GSW: ['#1D428A', '#FFC72C'], HOU: ['#CE1141', '#C4CED4'], IND: ['#002D62', '#FDBB30'],
  LAC: ['#C8102E', '#1D428A'], LAL: ['#552583', '#FDB927'], MEM: ['#5D76A9', '#12173F'],
  MIA: ['#98002E', '#F9A01B'], MIL: ['#00471B', '#EEE1C6'], MIN: ['#0C2340', '#236192'],
  NOP: ['#0C2340', '#C8102E'], NYK: ['#006BB6', '#F58426'], OKC: ['#007AC1', '#EF3B24'],
  ORL: ['#0077C0', '#C4CED4'], PHI: ['#006BB6', '#ED174C'], PHX: ['#1D1160', '#E56020'],
  POR: ['#E03A3E', '#1D1D1D'], SAC: ['#5A2D81', '#63727A'], SAS: ['#63727A', '#1D1D1D'],
  TOR: ['#CE1141', '#1D1D1D'], UTA: ['#002B5C', '#F9A01B'], WAS: ['#002B5C', '#E31837'],
};

const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const distance = (a, b) => Math.hypot(...rgb(a).map((v, i) => v - rgb(b)[i]));

// Away team switches to its secondary color when both primaries look alike
export const matchupColors = (awayAbbr, homeAbbr) => {
  const [awayPrimary, awaySecondary] = TEAM_COLORS[awayAbbr] || ['#64748B', '#94A3B8'];
  const [homePrimary] = TEAM_COLORS[homeAbbr] || ['#334155'];
  const away = distance(awayPrimary, homePrimary) < 90 ? awaySecondary : awayPrimary;
  return { away, home: homePrimary };
};

// "YYYY-MM-DD" in the viewer's local time (toISOString would shift to UTC)
export const isoLocal = (date) => {
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
};

export const dateFromIso = (iso) => {
  const [y, m, d] = String(iso).split('-').map(Number);
  return new Date(y, m - 1, d);
};

export const formatDate = (iso, opts = { weekday: 'long', month: 'long', day: 'numeric' }) =>
  dateFromIso(iso).toLocaleDateString('en-US', opts);

export const formatPct = (value, digits = 1) =>
  value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`;

const isTeam = (name, team) => !!name && (name === team.fullName || name.endsWith(team.name));

// Normalizes old and new JSON shapes into { homeProb (0-100), pickHome, confidence, ... }
export const normalizePrediction = (game) => {
  const p = game.prediction;
  if (!p || !p.winner || p.winner === 'TBD') return null;
  const pickHome = typeof p.predictedHomeWin === 'boolean' ? p.predictedHomeWin : isTeam(p.winner, game.homeTeam);
  const homeProb = typeof p.homeWinProbability === 'number'
    ? p.homeWinProbability
    : (pickHome ? p.confidence : 100 - p.confidence);
  return {
    pickHome,
    homeProb,
    confidence: p.confidence,
    correct: typeof p.correct === 'boolean' ? p.correct : null,
    keyFactors: p.keyFactors || [],
  };
};

export const formatStat = (value, format) => {
  if (value === null || value === undefined) return '—';
  if (format === 'frac') return `${(value * 100).toFixed(1)}%`;
  if (format === 'pct') return `${value.toFixed(1)}%`;
  return value.toFixed(1);
};

const fetchJson = async (name) => {
  const resp = await fetch(`${process.env.PUBLIC_URL}/data/${name}`, { cache: 'no-cache' });
  if (!resp.ok) throw new Error(`${name}: ${resp.status}`);
  return resp.json();
};

export const useSiteData = () => {
  const [state, setState] = useState({ loading: true, today: null, lastWeek: null, performance: null });
  useEffect(() => {
    Promise.allSettled([fetchJson('today.json'), fetchJson('lastweek.json'), fetchJson('performance.json')])
      .then(([today, lastWeek, performance]) => setState({
        loading: false,
        today: today.value || null,
        lastWeek: lastWeek.value || null,
        performance: performance.value || null,
      }));
  }, []);
  return state;
};

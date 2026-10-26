import React from 'react';
import { formatDate, formatPct } from '../data';

const Figure = ({ value, label, detail }) => (
  <div className="figure">
    <dt>{label}</dt>
    <dd>
      <span className="figure-value">{value}</span>
      {detail && <span className="figure-detail">{detail}</span>}
    </dd>
  </div>
);

// Bar = how often picks in this confidence range won; tick = what the model claimed on average
const Calibration = ({ buckets }) => (
  <div className="calib">
    <div className="calib-legend" aria-hidden="true">
      <span><i className="calib-key-bar" /> Actual win rate</span>
      <span><i className="calib-key-tick" /> Model's stated probability</span>
    </div>
    <table className="calib-table">
      <thead>
        <tr>
          <th scope="col">Stated probability</th>
          <th scope="col" className="calib-plot-head"><span className="sr-only">Chart</span></th>
          <th scope="col">Won</th>
          <th scope="col">Games</th>
        </tr>
      </thead>
      <tbody>
        {buckets.map((b) => (
          <tr key={b.range} tabIndex={0}
            title={b.games ? `${b.range}: stated ${b.avgConfidence}% on average, won ${formatPct(b.accuracy)} of ${b.games}` : undefined}>
            <th scope="row">{b.range}</th>
            <td className="calib-plot">
              {b.games > 0 && (
                <div className="calib-track">
                  <div className="calib-bar" style={{ width: `${b.accuracy * 100}%` }} />
                  <div className="calib-tick" style={{ left: `${b.avgConfidence}%` }} />
                </div>
              )}
            </td>
            <td>{b.games ? formatPct(b.accuracy, 0) : '—'}</td>
            <td>{b.games}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const ModelPage = ({ loading, performance }) => {
  if (loading) return <p className="page-status">Loading model stats…</p>;

  const overall = performance?.overall;
  const last7 = performance?.last7Days;
  const baseline = performance?.homeTeamBaseline;

  return (
    <>
      <section className="page-head">
        <h1>Track record</h1>
        <p className="page-sub">
          {performance?.trackingSince
            ? `Every pick is published before tip-off and graded after the final buzzer. Tracking since ${formatDate(performance.trackingSince, { month: 'long', day: 'numeric', year: 'numeric' })}.`
            : 'Every pick is published before tip-off and graded after the final buzzer.'}
        </p>
      </section>

      {overall?.games > 0 ? (
        <>
          <dl className="figures">
            <Figure label="All picks" value={formatPct(overall.accuracy)}
              detail={`${overall.correct}–${overall.games - overall.correct}`} />
            <Figure label="Last 7 days" value={last7?.games ? formatPct(last7.accuracy) : '—'}
              detail={last7?.games ? `${last7.correct}–${last7.games - last7.correct}` : 'No graded games'} />
            <Figure label="Always picking the home team" value={formatPct(baseline)}
              detail="Same games, for comparison" />
          </dl>

          <section className="panel">
            <h2>When the model says 70%, does it win 70% of the time?</h2>
            <p className="panel-sub">
              A well-calibrated model's bars reach its ticks. Bars short of the tick mean it was overconfident in that range.
            </p>
            <Calibration buckets={performance.byConfidence || []} />
          </section>
        </>
      ) : (
        <div className="empty"><p>No graded picks yet. Results appear here the morning after the first predicted games.</p></div>
      )}

      <section className="panel prose">
        <h2>How it works</h2>
        <p>
          Each team is described by its average box-score stats over its last 10 games: shooting efficiency,
          offensive and defensive rating, rebounding, turnovers, and its top performers. A gradient-boosted
          classifier compares the two teams and estimates the home team's chance of winning.
        </p>
        <p>
          The model scores each matchup from both teams' point of view and averages the two, so a game always
          gets one consistent probability. It uses the 50 stats, out of roughly 290 candidates, that
          mattered most in training.
        </p>
        <p>
          It was trained on regular season and playoff games from 2015 to 2021. On the 2021–22 season, which it never saw,
          it picked 61.9% of winners, against 54.7% for always picking the home team.
        </p>
      </section>
    </>
  );
};

export default ModelPage;

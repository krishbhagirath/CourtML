import React, { useState } from 'react';
import * as Logos from 'react-nba-logos';
import { formatDate, normalizePrediction } from '../data';

const dayRecord = (games) => {
  const graded = games.map(normalizePrediction).filter((p) => p && p.correct !== null);
  return { correct: graded.filter((p) => p.correct).length, total: graded.length };
};

const ScoreLine = ({ team, score, won, picked }) => {
  const Logo = Logos[team.abbreviation];
  return (
    <div className={`score-line ${won ? 'is-winner' : ''}`}>
      <span className="score-logo" aria-hidden="true">{Logo && <Logo size={28} />}</span>
      <span className="score-team">
        {team.city} {team.name}
        {picked && <span className="score-picked">Pick</span>}
      </span>
      <span className="score-pts">{score ?? '—'}</span>
    </div>
  );
};

const ResultCard = ({ game }) => {
  const pred = normalizePrediction(game);
  const { home, away } = game.finalScore || {};
  const homeWon = home > away;

  return (
    <li className="result">
      <ScoreLine team={game.awayTeam} score={away} won={!homeWon} picked={pred && !pred.pickHome} />
      <ScoreLine team={game.homeTeam} score={home} won={homeWon} picked={pred && pred.pickHome} />
      <p className={`result-verdict ${pred?.correct === true ? 'is-hit' : pred?.correct === false ? 'is-miss' : ''}`}>
        {!pred && 'No prediction made'}
        {pred && pred.correct === true && `Correct, ${pred.confidence.toFixed(0)}% pick`}
        {pred && pred.correct === false && `Missed, ${pred.confidence.toFixed(0)}% pick`}
        {pred && pred.correct === null && 'Awaiting result'}
      </p>
    </li>
  );
};

const ResultsPage = ({ loading, lastWeek }) => {
  const days = (lastWeek?.weekDates || []).map((date, i) => ({
    date,
    games: lastWeek.currentWeek[lastWeek.orderedDays[i]] || [],
  }));
  const latestWithGames = [...days].reverse().find((d) => d.games.length) || days[days.length - 1];
  const [selected, setSelected] = useState(null);
  const active = days.find((d) => d.date === selected) || latestWithGames;

  if (loading) return <p className="page-status">Loading results…</p>;
  if (!days.length) return <p className="page-status">Couldn't load recent results. Refresh the page to try again.</p>;

  const week = days.reduce((acc, d) => {
    const r = dayRecord(d.games);
    return { correct: acc.correct + r.correct, total: acc.total + r.total };
  }, { correct: 0, total: 0 });

  return (
    <>
      <section className="page-head">
        <h1>Results</h1>
        <p className="page-sub">
          {formatDate(days[0].date, { month: 'short', day: 'numeric' })} to{' '}
          {formatDate(days[days.length - 1].date, { month: 'short', day: 'numeric' })}
          {week.total > 0 && `. Picks went ${week.correct}–${week.total - week.correct} over these seven days.`}
        </p>
      </section>

      <div className="day-tabs" role="tablist" aria-label="Day">
        {days.map((d) => {
          const r = dayRecord(d.games);
          const isActive = d.date === active?.date;
          return (
            <button
              key={d.date}
              type="button"
              role="tab"
              aria-selected={isActive}
              className={`day-tab ${isActive ? 'is-active' : ''}`}
              onClick={() => setSelected(d.date)}
            >
              <span className="day-tab-name">{formatDate(d.date, { weekday: 'short' })}</span>
              <span className="day-tab-date">{formatDate(d.date, { month: 'short', day: 'numeric' })}</span>
              <span className="day-tab-rec">
                {r.total ? `${r.correct}–${r.total - r.correct}` : d.games.length ? `${d.games.length} games` : 'No games'}
              </span>
            </button>
          );
        })}
      </div>

      {active && active.games.length > 0 ? (
        <ul className="result-grid" role="tabpanel">
          {active.games.map((g) => <ResultCard key={g.id} game={g} />)}
        </ul>
      ) : (
        <div className="empty"><p>No games were played on this day.</p></div>
      )}
    </>
  );
};

export default ResultsPage;

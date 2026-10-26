import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import ProbabilityBar from './ProbabilityBar';
import TeamMark from './TeamMark';
import { formatDate, formatStat, isoLocal, normalizePrediction } from '../data';

const KeyFactors = ({ game, factors }) => (
  <table className="factors">
    <thead>
      <tr>
        <th scope="col">Last 10 games</th>
        <th scope="col">{game.awayTeam.abbreviation}</th>
        <th scope="col">{game.homeTeam.abbreviation}</th>
      </tr>
    </thead>
    <tbody>
      {factors.map((f) => (
        <tr key={f.label}>
          <th scope="row">
            {f.label}
            {!f.higherIsBetter && <span className="factor-hint"> (lower is better)</span>}
          </th>
          <td className={f.edge === 'away' ? 'is-edge' : ''}>{formatStat(f.away, f.format)}</td>
          <td className={f.edge === 'home' ? 'is-edge' : ''}>{formatStat(f.home, f.format)}</td>
        </tr>
      ))}
    </tbody>
  </table>
);

const GameRow = ({ game }) => {
  const [open, setOpen] = useState(false);
  const pred = normalizePrediction(game);
  const factorsId = `factors-${game.id}`;

  return (
    <li className="game">
      <div className="game-main">
        <TeamMark team={game.awayTeam} align="left" />
        <div className="game-center">
          <div className="game-when">
            {game.time}
            {game.venue && <span className="game-venue">{game.venue}</span>}
          </div>
          {pred ? (
            <ProbabilityBar awayTeam={game.awayTeam} homeTeam={game.homeTeam} homeProb={pred.homeProb} />
          ) : (
            <p className="game-nopick">No pick: not enough recent games for one team.</p>
          )}
        </div>
        <TeamMark team={game.homeTeam} align="right" />
      </div>

      {pred && (
        <div className="game-foot">
          <p className="game-pick">
            Pick: <strong>{pred.pickHome ? game.homeTeam.name : game.awayTeam.name}</strong>
            <span className="game-pick-conf"> {pred.confidence.toFixed(1)}% win probability</span>
          </p>
          {pred.keyFactors.length > 0 && (
            <button
              type="button"
              className="link-button"
              aria-expanded={open}
              aria-controls={factorsId}
              onClick={() => setOpen((o) => !o)}
            >
              {open ? 'Hide stats' : 'Compare stats'}
            </button>
          )}
        </div>
      )}
      {open && (
        <div id={factorsId} className="game-factors">
          <KeyFactors game={game} factors={pred.keyFactors} />
        </div>
      )}
    </li>
  );
};

const TonightPage = ({ loading, today }) => {
  if (loading) return <p className="page-status">Loading games…</p>;
  if (!today) return <p className="page-status">Couldn't load today's games. Refresh the page to try again.</p>;

  const isCurrent = today.date === isoLocal(new Date());
  const games = today.games || [];

  return (
    <>
      <section className="page-head">
        <h1>{isCurrent ? 'Tonight' : formatDate(today.date)}</h1>
        <p className="page-sub">
          {isCurrent && `${formatDate(today.date)}. `}
          {games.length} {games.length === 1 ? 'game' : 'games'}
        </p>
      </section>

      {!isCurrent && (
        <div className="notice" role="status">
          These are the latest predictions, from {formatDate(today.date, { month: 'long', day: 'numeric', year: 'numeric' })}.
          Today's games haven't been predicted yet.
        </div>
      )}

      {games.length > 0 ? (
        <ul className="game-list">
          {games.map((g) => <GameRow key={g.id} game={g} />)}
        </ul>
      ) : (
        <div className="empty">
          <p>No games on this date.</p>
          <Link to="/results">See recent results</Link>
        </div>
      )}
    </>
  );
};

export default TonightPage;

import React from 'react';
import { matchupColors } from '../data';

// Away share fills from the left, home share from the right
const ProbabilityBar = ({ awayTeam, homeTeam, homeProb, size = 'large' }) => {
  const awayProb = 100 - homeProb;
  const colors = matchupColors(awayTeam.abbreviation, homeTeam.abbreviation);
  const favorite = homeProb >= 50 ? 'home' : 'away';

  return (
    <div className={`prob prob-${size}`}>
      <div className="prob-labels">
        <span className={favorite === 'away' ? 'is-fav' : ''}>{awayProb.toFixed(0)}%</span>
        <span className={favorite === 'home' ? 'is-fav' : ''}>{homeProb.toFixed(0)}%</span>
      </div>
      <div
        className="prob-track"
        role="img"
        aria-label={`${awayTeam.name} ${awayProb.toFixed(0)} percent, ${homeTeam.name} ${homeProb.toFixed(0)} percent`}
      >
        <div className="prob-seg" style={{ width: `${awayProb}%`, background: colors.away }} />
        <div className="prob-seg" style={{ width: `${homeProb}%`, background: colors.home }} />
        <div className="prob-mid" />
      </div>
    </div>
  );
};

export default ProbabilityBar;

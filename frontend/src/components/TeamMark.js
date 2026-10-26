import React from 'react';
import * as Logos from 'react-nba-logos';

const TeamMark = ({ team, align = 'left', dim = false, children }) => {
  const Logo = Logos[team.abbreviation];
  return (
    <div className={`team-mark team-mark-${align} ${dim ? 'is-dim' : ''}`}>
      <div className="team-logo" aria-hidden="true">
        {Logo ? <Logo size={44} /> : <span className="team-logo-fallback">{team.abbreviation}</span>}
      </div>
      <div className="team-text">
        <span className="team-city">{team.city}</span>
        <span className="team-name">{team.name}</span>
        {children}
      </div>
    </div>
  );
};

export default TeamMark;

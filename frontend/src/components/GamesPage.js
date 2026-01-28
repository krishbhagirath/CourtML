import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  LAL, GSW, BOS, MIA, CHI, MIL, BKN, NYK, PHX, LAC, DAL, HOU, POR, UTA,
  ATL, CHA, CLE, DEN, DET, IND, MEM, MIN, NOP, OKC, ORL, PHI, SAC, SAS, TOR, WAS
} from 'react-nba-logos';
import PredictionChart from './PredictionChart';
import './GamesPage.css';

// "7:30p" -> minutes since midnight; unknown -> very large so it sorts last
// "7:30p" -> minutes since midnight; unknown -> very large so it sorts last
// const parseTimeToMinutes = (t) => { ... } // Removed unused function

// Convert "YYYY-MM-DD" to Date at midnight (local)
const dateFromISOLocal = (iso) => {
  const [y, m, d] = String(iso).split('-').map(Number);
  return new Date(y, m - 1, d);
};

// Fallback Mon→Sun if JSON missing
const buildFallbackWeekDates = () => {
  const today = new Date();
  const res = [];
  const dow = today.getDay(); // 0=Sun
  const mondayOffset = dow === 0 ? -6 : 1 - dow;
  const monday = new Date(today);
  monday.setDate(today.getDate() + mondayOffset);
  for (let i = 0; i < 7; i++) {
    const d = new Date(monday);
    d.setDate(monday.getDate() + i);
    res.push(d);
  }
  return res;
};

const GamesPage = () => {
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [viewMode, setViewMode] = useState('current'); // 'current' or 'lastWeek'
  const [slideDirection, setSlideDirection] = useState('');
  const [todayData, setTodayData] = useState(null);
  const [lastWeekData, setLastWeekData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedGames, setExpandedGames] = useState(new Set());

  // Load today.json and lastweek.json
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);

        // 1. Fetch Today's Games
        try {
          const todayResp = await fetch('/data/today.json');
          if (todayResp.ok) {
            const tData = await todayResp.json();
            setTodayData(tData);
            // Default selected date to today
            if (tData.date) {
              setSelectedDate(dateFromISOLocal(tData.date));
            }
          }
        } catch (e) {
          console.error("Failed to fetch today.json", e);
        }

        // 2. Fetch Last Week's Games
        try {
          const lastWeekResp = await fetch('/data/lastweek.json');
          if (lastWeekResp.ok) {
            const lData = await lastWeekResp.json();
            setLastWeekData(lData);
          }
        } catch (e) {
          console.error("Failed to fetch lastweek.json", e);
        }

      } catch (error) {
        console.error('Error loading games data:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Team logo mapping
  const teamLogos = {
    'Lakers': LAL, 'Warriors': GSW, 'Celtics': BOS, 'Heat': MIA, 'Bulls': CHI, 'Bucks': MIL,
    'Nets': BKN, 'Knicks': NYK, 'Suns': PHX, 'Clippers': LAC, 'Mavericks': DAL, 'Rockets': HOU,
    'Trail Blazers': POR, 'Jazz': UTA, 'Hawks': ATL, 'Hornets': CHA, 'Cavaliers': CLE, 'Nuggets': DEN,
    'Pistons': DET, 'Pacers': IND, 'Grizzlies': MEM, 'Timberwolves': MIN, 'Pelicans': NOP, 'Thunder': OKC,
    'Magic': ORL, '76ers': PHI, 'Kings': SAC, 'Spurs': SAS, 'Raptors': TOR, 'Wizards': WAS
  };

  // Get all season dates or fallback
  const getWeekDates = () => {
    if (viewMode === 'current') {
      if (todayData?.date) {
        // Just return today as a single item array? Or pad it?
        // User said: "use the current date to fetch todays games... then compute dates of the last 7 days... lay them out as dates"
        // It implies for "current" view maybe just today is fine, or maybe surrounding days?
        // Let's stick to just Today for now to be safe with available data.
        return [dateFromISOLocal(todayData.date)];
      }
      return [new Date()]; // fallback
    } else {
      // Last Week
      if (lastWeekData?.weekDates) {
        return lastWeekData.weekDates.map(d => dateFromISOLocal(d));
      }
      return buildFallbackWeekDates();
    }
  };

  const weekDates = getWeekDates();

  const handleDateSelect = (date) => setSelectedDate(date);

  const switchToLastWeek = () => {
    setSlideDirection('left');
    // If we have last week data, select the first day or last day?
    // Usually last day (closest to today) is better, or first day?
    // Let's default to the last day of the week (yesterday)
    if (lastWeekData?.weekDates?.length) {
      const lastDay = lastWeekData.weekDates[lastWeekData.weekDates.length - 1];
      setSelectedDate(dateFromISOLocal(lastDay));
    }

    setTimeout(() => {
      setViewMode('lastWeek');
      setSlideDirection('right');
    }, 300);
  };

  const switchToCurrentWeek = () => {
    setSlideDirection('right');
    // Select today
    if (todayData?.date) {
      setSelectedDate(dateFromISOLocal(todayData.date));
    }
    setTimeout(() => {
      setViewMode('current');
      setSlideDirection('left');
    }, 300);
  };

  // Get games for selected date
  const getSelectedDayGames = () => {
    const isoDate = selectedDate.toISOString().split('T')[0]; // YYYY-MM-DD

    if (viewMode === 'current') {
      if (todayData?.date === isoDate) {
        return todayData.games || [];
      }
      return [];
    } else {
      // Last Week
      // lastWeekData has currentWeek keyed by Day Name (e.g. "Monday")
      // We need to find the day name for the isoDate
      if (lastWeekData?.weekDates && lastWeekData?.orderedDays) {
        const idx = lastWeekData.weekDates.indexOf(isoDate);
        if (idx !== -1) {
          const dayName = lastWeekData.orderedDays[idx];
          return lastWeekData.currentWeek[dayName] || [];
        }
      }
      return [];
    }
  };

  const getTeamLogo = (teamName) => {
    const LogoComponent = teamLogos[teamName];
    return LogoComponent ? <LogoComponent size={40} /> : <div className="team-logo-placeholder">{teamName?.charAt(0)}</div>;
  };

  const toggleGameStats = (gameId) => {
    setExpandedGames(prev => {
      const newSet = new Set(prev);
      if (newSet.has(gameId)) {
        newSet.delete(gameId);
      } else {
        newSet.add(gameId);
      }
      return newSet;
    });
  };


  return (
    <div
      className="games-page"
      style={{
        backgroundImage: `linear-gradient(to bottom, rgba(15, 23, 42, 0.85), rgba(0, 0, 0, 0.95)), url(${process.env.PUBLIC_URL}/bg_player.png)`
      }}
    >
      {/* Navigation */}
      <nav className="nav">
        <div className="nav-container">
          <Link to="/" className="nav-logo">
            <img
              src="https://upload.wikimedia.org/wikipedia/en/thumb/0/03/National_Basketball_Association_logo.svg/451px-National_Basketball_Association_logo.svg.png"
              alt="NBA Logo"
              style={{ height: '40px', verticalAlign: 'middle', marginRight: '8px' }}
            />
            CourtML
          </Link>
          <div className="nav-links">
            <Link to="/" className="nav-link">Home</Link>
            <Link to="/games" className="nav-link">Games & Predictions</Link>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="games-container">
        <div className="games-header">
          <h1>Games & Predictions</h1>
          <p>View upcoming games and past prediction results</p>
        </div>

        {/* Calendar Navigation */}
        <div className="calendar-navigation">
          {viewMode === 'current' && (
            <button className="nav-button last-week-btn" onClick={switchToLastWeek}>
              <span className="arrow">←</span>
              Last Week's Predictions
            </button>
          )}
          {viewMode === 'lastWeek' && (
            <button className="nav-button current-week-btn" onClick={switchToCurrentWeek}>
              This Week's Games
              <span className="arrow">→</span>
            </button>
          )}
        </div>

        {/* Current Week */}
        {viewMode === 'current' && (
          <div className={`calendar-week ${slideDirection === 'left' ? 'slide-left' : slideDirection === 'right' ? 'slide-right' : ''}`}>
            <div className="week-header">
              <h2>Season Games</h2>
              <p style={{ marginBottom: '24px' }}>Today's games with AI-powered predictions</p>
            </div>

            <div className="day-bubbles">
              {weekDates.map((date, index) => {
                const isoDate = date.toISOString().split('T')[0];
                const games = (todayData?.date === isoDate) ? (todayData.games || []) : [];
                const isSelected = date.toDateString() === selectedDate.toDateString();
                const isToday = date.toDateString() === new Date().toDateString();

                return (
                  <div
                    key={index}
                    className={`day-bubble ${isSelected ? 'selected' : ''} ${isToday ? 'today' : ''}`}
                    onClick={() => handleDateSelect(date)}
                  >
                    <div className="day-name">{date.toLocaleDateString('en-US', { weekday: 'short' })}</div>
                    <div className="day-date">{date.getDate()}</div>
                    <div className="games-count">{games.length} game{games.length !== 1 ? 's' : ''}</div>
                  </div>
                );
              })}
            </div>

            {/* Selected Day Games */}
            <div className="selected-day-games">
              <h3 className="selected-day-title">
                {selectedDate.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
              </h3>

              {loading ? (
                <div className="no-games"><p>Loading…</p></div>
              ) : getSelectedDayGames().length > 0 ? (
                <div className="games-list" key={selectedDate.toISOString()}>
                  {getSelectedDayGames().map(game => (
                    <div key={game.id} className="game-card prediction-card">
                      <div className="game-teams">
                        {/* Home Team: Name Left, Logo Right */}
                        <div className="team-info home-team">
                          <span className="team-name">{game.homeTeam.name}</span>
                          <div className="team-logo-container">{getTeamLogo(game.homeTeam.name)}</div>
                        </div>

                        <span className="vs-text">VS</span>

                        {/* Away Team: Logo Left, Name Right */}
                        <div className="team-info away-team">
                          <div className="team-logo-container">{getTeamLogo(game.awayTeam.name)}</div>
                          <span className="team-name">{game.awayTeam.name}</span>
                        </div>
                      </div>

                      {/* Always show time/venue for today's games (updates once daily) */}
                      <div className="game-details">
                        <div className="game-time">{game.time}</div>
                        {game.venue && (
                          <>
                            <span className="separator">•</span>
                            <div className="game-venue">{game.venue}</div>
                          </>
                        )}
                      </div>

                      <div className="prediction-section">
                        {game.prediction && game.prediction.winner !== 'TBD' ? (
                          <>
                            {game.prediction.correct !== undefined ? (
                              /* Result View (Past Games) */
                              <div className="prediction-result-container">
                                <div className="prediction">
                                  <strong>Predicted:</strong>
                                  <span className="prediction-winner-name">{game.prediction.winner}</span>
                                </div>
                                <div className={`result-badge ${game.prediction.correct ? 'correct' : 'incorrect'}`}>
                                  {game.prediction.correct ? '✅ Correct' : '❌ Incorrect'}
                                </div>
                              </div>
                            ) : (
                              /* Live Prediction View */
                              <>
                                <div className="prediction">
                                  <strong>PREDICTION:</strong>
                                  <span className="prediction-winner-name">{game.prediction.winner.toUpperCase()} WINS</span>
                                </div>
                                <div className="confidence-bar">
                                  <div
                                    className="confidence-fill"
                                    style={{ width: `${game.prediction.confidence}%` }}
                                  />
                                </div>
                                <div className="confidence-value">{game.prediction.confidence}% Confidence</div>

                                {/* Stats Toggle Button */}
                                {game.prediction.keyDifferences && (
                                  <>
                                    <button
                                      className={`stats-toggle-btn ${expandedGames.has(game.id) ? 'expanded' : ''}`}
                                      onClick={() => toggleGameStats(game.id)}
                                    >
                                      <span>{expandedGames.has(game.id) ? 'Hide' : 'View'} Stats Breakdown</span>
                                      <span className="arrow">▼</span>
                                    </button>

                                    {/* Stats Chart */}
                                    {expandedGames.has(game.id) && (
                                      <PredictionChart
                                        homeTeam={game.homeTeam.name}
                                        awayTeam={game.awayTeam.name}
                                        keyDifferences={game.prediction.keyDifferences}
                                      />
                                    )}
                                  </>
                                )}
                              </>
                            )}
                          </>
                        ) : (
                          <div className="prediction-scheduled">
                            Scheduled
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-games"><p>No games scheduled for this day</p></div>
              )}
            </div>
          </div>
        )}

        {/* Last Week */}
        {viewMode === 'lastWeek' && (
          <div className={`calendar-week ${slideDirection === 'right' ? 'slide-right' : slideDirection === 'left' ? 'slide-left' : ''}`}>
            <div className="week-header">
              <h2>Last Week's Results</h2>
              <p style={{ marginBottom: '24px' }}>Review past predictions and actual game outcomes</p>
            </div>

            <div className="day-bubbles">
              {weekDates.map((date, index) => {
                const isoDate = date.toISOString().split('T')[0];
                const dayGames = (viewMode === 'lastWeek' && lastWeekData?.weekDates && lastWeekData?.orderedDays)
                  ? (() => {
                    const idx = lastWeekData.weekDates.indexOf(isoDate);
                    if (idx !== -1) {
                      const dayName = lastWeekData.orderedDays[idx];
                      return lastWeekData.currentWeek[dayName] || [];
                    }
                    return [];
                  })()
                  : [];
                const isSelected = date.toDateString() === selectedDate.toDateString();

                return (
                  <div
                    key={index}
                    className={`day-bubble ${isSelected ? 'selected' : ''}`}
                    onClick={() => handleDateSelect(date)}
                  >
                    <div className="day-name">{date.toLocaleDateString('en-US', { weekday: 'short' })}</div>
                    <div className="day-date">{date.getDate()}</div>
                    <div className="games-count">{dayGames.length} game{dayGames.length !== 1 ? 's' : ''}</div>
                  </div>
                );
              })}
            </div>

            {/* Selected Day Games */}
            <div className="selected-day-games">
              <h3 className="selected-day-title">
                {selectedDate.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
              </h3>

              {loading ? (
                <div className="no-games"><p>Loading…</p></div>
              ) : getSelectedDayGames().length > 0 ? (
                <div className="games-list" key={selectedDate.toISOString()}>
                  {getSelectedDayGames().map(game => (
                    <div key={game.id} className="game-card prediction-card">
                      <div className="game-teams">
                        {/* Home Team: Name Left, Logo Right */}
                        <div className="team-info home-team">
                          <span className="team-name">{game.homeTeam.name}</span>
                          <div className="team-logo-container">{getTeamLogo(game.homeTeam.name)}</div>
                        </div>

                        <span className="vs-text">VS</span>

                        {/* Away Team: Logo Left, Name Right */}
                        <div className="team-info away-team">
                          <div className="team-logo-container">{getTeamLogo(game.awayTeam.name)}</div>
                          <span className="team-name">{game.awayTeam.name}</span>
                        </div>
                      </div>

                      {/* Final Score for Last Week */}
                      {game.finalScore && (game.finalScore.home || game.finalScore.away) ? (
                        <>
                          <div className="score-display">
                            <div className="team-score">
                              <span className="score-team-abbr">{game.homeTeam.abbreviation}</span>
                              <span className="score-number">{game.finalScore.home}</span>
                            </div>
                            <span className="score-separator">-</span>
                            <div className="team-score">
                              <span className="score-number">{game.finalScore.away}</span>
                              <span className="score-team-abbr">{game.awayTeam.abbreviation}</span>
                            </div>
                          </div>
                          <div className="game-status-label">{game.time || game.gameStatus}</div>
                        </>
                      ) : (
                        <div className="game-details">
                          <div className="game-time">{game.time || 'TBD'}</div>
                          {game.venue && (
                            <>
                              <span className="separator">•</span>
                              <div className="game-venue">{game.venue}</div>
                            </>
                          )}
                        </div>
                      )}

                      <div className="prediction-section">
                        {game.prediction && game.prediction.winner ? (
                          <div className="prediction-result-container">
                            <div className="prediction">
                              <strong>Predicted:</strong>
                              <span className="prediction-winner-name">{game.prediction.winner}</span>
                            </div>
                            {game.prediction.correct !== undefined && (
                              <div className={`result-badge ${game.prediction.correct ? 'correct' : 'incorrect'}`}>
                                {game.prediction.correct ? '✅ Correct' : '❌ Incorrect'}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="prediction-scheduled">
                            No Prediction Available
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-games"><p>No games played on this day</p></div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GamesPage;

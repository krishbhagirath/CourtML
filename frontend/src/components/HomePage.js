import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';

const HomePage = () => {
  return (
    <div
      className="home-page"
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
              style={{ height: '32px', verticalAlign: 'middle' }}
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
      <div className="home-container">
        {/* Left side: Text content */}
        <div className="home-content-left">
          {/* Logo + text side-by-side */}
          <div className="home-header">
            <div className="logo-title-group">
              <img
                className="brand-logo"
                src="https://upload.wikimedia.org/wikipedia/en/thumb/0/03/National_Basketball_Association_logo.svg/451px-National_Basketball_Association_logo.svg.png"
                alt="NBA Logo"
              />
              <h1 className="project-title">CourtML</h1>
            </div>

            <p className="project-subtitle">
              Advanced Machine Learning for NBA Live Predictions
            </p>
          </div>

          {/* Call to Action Button */}
          <Link to="/games" className="cta-button">
            View Games
          </Link>

          {/* Additional Info */}
          <div className="project-info">
            <p>
              Powered by HistGradientBoosting & Python to analyze 50+ stats per game.
            </p>
          </div>
        </div>

        {/* Right side: Background clean image */}
        <div className="home-content-right">
          <img
            src={`${process.env.PUBLIC_URL}/background_clean.png`}
            alt="NBA Player"
            className="hero-player-image"
          />
        </div>
      </div>
    </div>
  );
};

export default HomePage; 
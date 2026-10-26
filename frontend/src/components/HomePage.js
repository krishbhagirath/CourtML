import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => (
  <section className="hero">
    <div className="hero-copy">
      <h1 className="hero-title">CourtML</h1>
      <p className="hero-lede">
        Machine-learning predictions for every NBA game, built from each team's last 10 games,
        with every pick graded in public.
      </p>
      <div className="hero-actions">
        <Link to="/games" className="button-primary">See tonight's picks</Link>
        <Link to="/results" className="button-quiet">Recent results</Link>
      </div>
    </div>
    <div className="hero-art">
      <img
        src={`${process.env.PUBLIC_URL}/background_clean.png`}
        alt="LeBron James dribbling in a Cleveland Cavaliers jersey"
        className="hero-player"
        width="418"
        height="597"
      />
    </div>
  </section>
);

export default HomePage;

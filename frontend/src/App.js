import React from 'react';
import { BrowserRouter as Router, Navigate, NavLink, Route, Routes } from 'react-router-dom';
import './App.css';
import { useSiteData } from './data';
import TonightPage from './components/TonightPage';
import ResultsPage from './components/ResultsPage';
import ModelPage from './components/ModelPage';

function App() {
  const data = useSiteData();

  return (
    <Router>
      <header className="site-header">
        <div className="shell site-header-inner">
          <NavLink to="/" className="wordmark" aria-label="CourtML home">
            Court<span>ML</span>
          </NavLink>
          <nav className="site-nav" aria-label="Main">
            <NavLink to="/" end>Tonight</NavLink>
            <NavLink to="/results">Results</NavLink>
            <NavLink to="/model">Track record</NavLink>
          </nav>
        </div>
      </header>

      <main className="shell">
        <Routes>
          <Route path="/" element={<TonightPage {...data} />} />
          <Route path="/results" element={<ResultsPage {...data} />} />
          <Route path="/model" element={<ModelPage {...data} />} />
          <Route path="/games" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="shell site-footer">
        Predictions are model estimates from each team's last 10 games. Not betting advice.
      </footer>
    </Router>
  );
}

export default App;

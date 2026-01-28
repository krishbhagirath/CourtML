import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';
import './PredictionChart.css';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const PredictionChart = ({ homeTeam, awayTeam, keyDifferences }) => {
    if (!keyDifferences || keyDifferences.length === 0) {
        return null;
    }

    // Extract data for chart
    const labels = keyDifferences.map(f => f.name);
    const homeValues = keyDifferences.map(f => f.homeValue);
    const awayValues = keyDifferences.map(f => f.awayValue);

    const data = {
        labels: labels,
        datasets: [
            {
                label: homeTeam,
                data: homeValues,
                backgroundColor: 'rgba(59, 130, 246, 0.8)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1,
            },
            {
                label: awayTeam,
                data: awayValues,
                backgroundColor: 'rgba(239, 68, 68, 0.8)',
                borderColor: 'rgba(239, 68, 68, 1)',
                borderWidth: 1,
            },
        ],
    };

    const options = {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.5,
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                },
                ticks: {
                    color: '#94a3b8',
                    font: {
                        size: 11,
                    },
                },
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)',
                },
                ticks: {
                    color: '#e2e8f0',
                    font: {
                        size: 12,
                        weight: 'bold',
                    },
                },
            },
        },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#e2e8f0',
                    padding: 15,
                    font: {
                        size: 13,
                        weight: 'bold',
                    },
                },
            },
            tooltip: {
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                titleColor: '#e2e8f0',
                bodyColor: '#cbd5e1',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
                padding: 12,
                bodyFont: {
                    size: 13,
                },
                callbacks: {
                    afterLabel: function (context) {
                        const index = context.dataIndex;
                        const diff = keyDifferences[index];
                        const advantage = diff.homeAdvantage ? homeTeam : awayTeam;
                        return `Difference: ${diff.difference} (${advantage} advantage)`;
                    }
                }
            },
        },
    };

    return (
        <div className="prediction-chart-container">
            <h4 className="chart-title">Top Differentiating Stats</h4>
            <p className="chart-subtitle">Features that differ most between teams for this matchup</p>
            <div className="chart-wrapper">
                <Bar data={data} options={options} />
            </div>
            <p className="chart-note">
                Based on last 10 games rolling average
            </p>
        </div>
    );
};

export default PredictionChart;

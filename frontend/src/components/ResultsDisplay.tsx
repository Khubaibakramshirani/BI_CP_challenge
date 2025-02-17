// src/components/ResultsDisplay.tsx
import React from 'react';

interface ResultsDisplayProps {
    result: string;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result }) => {
    return (
        <div className="results-display">
            <h2>Response:</h2>
            <p>{result}</p>
        </div>
    );
};

export default ResultsDisplay;

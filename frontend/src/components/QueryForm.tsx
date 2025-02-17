// src/components/QueryForm.tsx
import React, { useState } from 'react';

interface QueryFormProps {
    onSubmit: (question: string) => void;
}

const QueryForm: React.FC<QueryFormProps> = ({ onSubmit }) => {
    const [question, setQuestion] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSubmit(question);
    };

    return (
        <form onSubmit={handleSubmit} className="query-form">
            <input
                type="text"
                placeholder="Ask a question..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="query-input"
            />
            <button type="submit" className="query-submit">Submit</button>
        </form>
    );
};

export default QueryForm;

import React, { useState } from 'react';
import './App.css';
import { BASE_API_URL } from './constants';

interface Message {
    text: string;
    sender: 'user' | 'bot';
}

const App: React.FC = () => {
    const [question, setQuestion] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [documentType, setDocumentType] = useState('presentation'); // Default to presentation

    const handleQuerySubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!question.trim()) return;
        
        setMessages(prevMessages => [...prevMessages, { text: question, sender: 'user' }]);
        setLoading(true);
        setError('');

        try {
            const response = await fetch(`${BASE_API_URL}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    question,
                    documentType // Include document type in the request
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            setMessages(prevMessages => [
                ...prevMessages,
                { text: data.response, sender: 'bot' }
            ]);

            if (data.response?.toLowerCase() === "i don't know.") {
                setMessages(prevMessages => [
                    ...prevMessages,
                    {
                        text: "Try rephrasing your question or using prompt engineering to improve clarity.",
                        sender: 'bot'
                    }
                ]);
            }
        } catch (err) {
            console.error('Error:', err);
            setError('Failed to fetch response. Please try again later.');
        } finally {
            setLoading(false);
            setQuestion('');
        }
    };

    return (
        <div className="app-container">
            <h1 className="chat-heading">
                RAG Chatbot: ConocoPhillips
            </h1>
            <div className="document-selector">
                <select 
                    value={documentType}
                    onChange={(e) => setDocumentType(e.target.value)}
                    className="document-select"
                >
                    <option value="presentation">Presentation</option>
                    <option value="proxy_statement">Proxy Statement</option>
                </select>
            </div>
            <div className="chat-box">
                {messages.map((message, index) => (
                    <div key={index} className={`chat-bubble ${message.sender}`}>
                        <p>{message.text}</p>
                    </div>
                ))}
                {loading && <div className="loading">Thinking...</div>}
            </div>
            <form onSubmit={handleQuerySubmit} className="input-area">
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Type your question..."
                />
                <button type="submit" disabled={loading}>Send</button>
            </form>
            {error && <p className="error-message">{error}</p>}
        </div>
    );
};

export default App;
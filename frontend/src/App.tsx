import React, { useState, useRef, KeyboardEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';
import { BASE_API_URL } from './constants';

interface Message {
    text: string;
    sender: 'user' | 'bot';
    context?: {
        texts: Array<{
            text: string;
            metadata: any;
        }>;
        images: string[];  // base64 encoded images
    };
}

// Helper function to format math expressions in text
const formatMathExpressions = (text: string) => {
    // Replace LaTeX-style expressions with more readable format
    return text.replace(/\[ \\text\{(.+?)\} \]/g, '$1')
              .replace(/\\text\{(.+?)\}/g, '$1');
};

// New component for displaying messages and their context
const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
    // Function to determine if images should be rendered
    const shouldRenderImages = () => {
        const noInfoKeywords = [
            "i don't know",
            "i'm unable to determine",
            "i cannot find",
            "i do not know",
            "there is no information",
            "not enough context",
            "based solely on the context you provided",
            "unable to provide specific details",
            "cannot determine",
            "can't determine"
        ];

        return !(
            noInfoKeywords.some(keyword => 
                message.text.toLowerCase().includes(keyword)
            )
        );
    };

    // Pre-process the text to handle any math expressions
    const processedText = message.sender === 'bot' ? formatMathExpressions(message.text) : message.text;

    return (
        <div className={`chat-bubble ${message.sender}`}>
            {message.sender === 'user' ? (
                <p>{message.text}</p>
            ) : (
                <div className="markdown-content">
                    <ReactMarkdown>{processedText}</ReactMarkdown>
                </div>
            )}
            {shouldRenderImages() && 
             message.context && 
             message.context.images && 
             message.context.images.length > 0 && (
                <div className="context-container">
                    {message.context.images.map((base64Image, index) => (
                        <div key={index} className="image-container">
                            <img 
                                src={`data:image/jpeg;base64,${base64Image}`}
                                alt={`Reference ${index + 1}`}
                                className="context-image"
                            />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

const App: React.FC = () => {
    const [question, setQuestion] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [documentType, setDocumentType] = useState('presentation');
    const textareaRef = useRef<HTMLTextAreaElement>(null);

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
                    documentType
                }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
           
            setMessages(prevMessages => [
                ...prevMessages,
                { 
                    text: data.response, 
                    sender: 'bot',
                    context: data.context // Include the context with images
                }
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
            if (textareaRef.current) {
                textareaRef.current.style.height = 'auto';
            }
        }
    };

    const handleQuestionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setQuestion(e.target.value);
        
        // Auto-resize textarea
        const textarea = e.target;
        textarea.style.height = 'auto';
        textarea.style.height = `${textarea.scrollHeight}px`;
    };

    // Handle key press events in the textarea
    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        // If user presses Shift+Enter, insert a newline instead of submitting
        if (e.key === 'Enter' && e.shiftKey) {
            e.preventDefault();
            const cursorPosition = e.currentTarget.selectionStart;
            const textBeforeCursor = question.substring(0, cursorPosition);
            const textAfterCursor = question.substring(cursorPosition);
            
            setQuestion(textBeforeCursor + '\n' + textAfterCursor);
            
            // Set cursor position after the inserted newline
            setTimeout(() => {
                if (textareaRef.current) {
                    textareaRef.current.selectionStart = cursorPosition + 1;
                    textareaRef.current.selectionEnd = cursorPosition + 1;
                }
            }, 0);
        } else if (e.key === 'Enter' && !e.shiftKey) {
            // Submit the form on plain Enter
            e.preventDefault();
            handleQuerySubmit(e);
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
                    <MessageBubble key={index} message={message} />
                ))}
                {loading && <div className="loading">Thinking...</div>}
            </div>
            <form onSubmit={handleQuerySubmit} className="input-area">
                <textarea
                    ref={textareaRef}
                    value={question}
                    onChange={handleQuestionChange}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your question... (Shift+Enter for new line)"
                    rows={1}
                    className="question-textarea"
                />
                <button type="submit" disabled={loading}>Send</button>
            </form>
            {error && <p className="error-message">{error}</p>}
        </div>
    );
};

export default App;
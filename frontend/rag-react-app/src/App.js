import React, { useState, useEffect } from 'react';

function App() {
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState([]);

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws');

        ws.onmessage = (event) => {
            console.log('Received message:', event.data);
            setResponse(prev => [...prev, event.data]);
        };
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
        };
        return () => {
            ws.close();
        };
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        const ws = new WebSocket('ws://localhost:8000/ws');
        ws.onopen = () => ws.send(query);
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder="Ask your question"
                />
                <button type="submit">Submit</button>
            </form>
            <div>
                <h3>Responses:</h3>
                {response.map((res, index) => <p key={index}>{res}</p>)}
            </div>
        </div>
    );
}

export default App;
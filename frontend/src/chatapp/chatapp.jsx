import React, { useState, useEffect } from 'react';

function ChatPage() {
    const [messages, setMessages] = useState([
        { text: 'Hello! How can I help you?', type: 'received' },
        { text: 'Hi, I need some information.', type: 'sent' }
    ]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState('');

    useEffect(() => {
        const fetchSessionId = async () => {
            const response = await fetch("http://localhost:8000/new-session");
            const data = await response.json();
            setSessionId(data.session_id);  // Store session ID
        };

        fetchSessionId();
    }, []);

    const sendMessage = async () => {
        if (input.trim() !== '') {
            setMessages((prevMessages) => [...prevMessages, { text: input, type: 'sent' }]);
            setInput('');

            const response = await fetch("http://localhost:8000/rag-query", {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,  // Send the session ID with each query
                    query: input
                })
            });

            const data = await response.json();
            setMessages((prevMessages) => [...prevMessages, { text: data.response, type: 'received' }]);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-screen w-screen border border-gray-300">
            <div className="bg-emerald-500 text-white p-4 text-center text-lg">
                KUTCAS connect
            </div>
            <div className="flex-grow p-4 overflow-y-auto bg-gray-100">
                {messages.map((msg, index) => (
                    <div key={index} className={`flex items-center mb-3 ${msg.type === 'sent' ? 'justify-end' : 'justify-start'}`}>
                        {msg.type === 'received' && (
                            <div className="w-8 h-8 bg-gray-400 rounded-full flex items-center justify-center mr-2">
                                B
                            </div>
                        )}
                        <div
                            className={`p-3 rounded-large max-w-xs ${msg.type === 'sent' ? 'bg-emerald-500 text-white' : 'bg-gray-300 text-black'
                                }`}
                        >
                            {msg.text}
                        </div>
                        {msg.type === 'sent' && (
                            <div className="w-8 h-8 bg-emerald-600 rounded-full flex items-center justify-center ml-2 text-white">
                                U
                            </div>
                        )}
                    </div>
                ))}
            </div>
            <div className="flex p-4 border-t border-gray-300 bg-white">
                <input
                    type="text"
                    className="flex-grow p-2 border border-gray-300 rounded-lg mr-2"
                    placeholder="Type a message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyPress}
                />
                <button
                    className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
                    onClick={sendMessage}
                >
                    Send
                </button>
            </div>
        </div>
    );
}

export default ChatPage;

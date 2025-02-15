import { useState, useEffect, useRef } from 'react';
import Navbar from '../components/navbar';
import { Icon } from "@iconify/react";
import { BeatLoader, ClipLoader } from "react-spinners";  // Import the spinner

function ChatPage() {
    const [messages, setMessages] = useState([
        { text: 'Hello! How can I help you?', type: 'received' },
        { text: 'Hi, I need some information.', type: 'sent' },
    ]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [loading, setLoading] = useState(false);
    const lastMessageRef = useRef(null);  // Reference to the last message

    useEffect(() => {
        const fetchSessionId = async () => {
            const response = await fetch("http://localhost:8000/new-session");
            const data = await response.json();
            setSessionId(data.session_id);  // Store session ID
        };

        fetchSessionId();
    }, []);

    useEffect(() => {
        if (lastMessageRef.current) {
            lastMessageRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);  // Scroll to the latest message when messages change

    const sendMessage = async () => {
        if (input.trim() !== '') {
            setMessages((prevMessages) => [...prevMessages, { text: input, type: 'sent' }]);
            setInput('');
            setLoading(true);

            // Add a loading message for the response
            setMessages((prevMessages) => [...prevMessages, { text: 'Loading...', type: 'received', loading: true }]);

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
            setMessages((prevMessages) => {
                // Replace the "Loading..." message with the actual response
                const updatedMessages = [...prevMessages];
                updatedMessages[updatedMessages.length - 1] = { text: data.response, type: 'received' };
                return updatedMessages;
            });
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-screen w-screen border border-gray-300">
            <Navbar />
            <div className='mx-8 mt-4 py-4 px-8 border border-gray-300 rounded-lg rounded-b-none'>
                <span className='flex items-center space-x-2'>
                    <Icon icon="fluent:bot-20-filled" width="32" height="32" style={{ color: "#0097B2", background:"#D9EBEE", borderRadius:"50%"}} />
                    <p className='px-1 font-medium'>น้องคอนเนค</p>
                </span>
            </div>
            
            <div className="flex-grow mx-8 mb-4 p-4 overflow-y-auto border border-gray-300 rounded-lg shadow-lg rounded-t-none">
                {messages.map((msg, index) => (
                    <div key={index} className={`flex items-center mb-3 ${msg.type === 'sent' ? 'justify-end' : 'justify-start'}`}>
                        {msg.type === 'received' && !msg.loading && (
                            <div className="w-8 h-8 bg-gray-400 rounded-full flex items-center justify-center mr-2 self-start">
                                <Icon icon="fluent:bot-20-filled" width="32" height="32" style={{ color: "#0097B2", background:"#D9EBEE", borderRadius:"50%"}} />
                            </div>
                        )}
                        <div
                            className={`p-2 rounded-large max-w-xs ${msg.type === 'sent' ? 'bg-kutcasgreen100 text-black rounded-lg rounded-br-none' : 'bg-gray-300 bg-kutcasgreen100 text-black rounded-t-lg rounded-r-lg rounded-bl-none'}`}
                        >
                            {msg.loading ? (
                                <div className="flex justify-center items-center">
                                    <BeatLoader color="#0097B2" size={12} />
                                </div>
                            ) : (
                                msg.text
                            )}
                        </div>
                        {msg.type === 'sent' && (
                            <div className="w-8 h-8 bg-kutcasgreen700 rounded-full flex items-center justify-center ml-2 text-white">
                                U
                            </div>
                        )}
                    </div>
                ))}
                {/* This element ensures we always scroll to the latest message */}
                <div ref={lastMessageRef} />
            </div>
            <div className="flex p-4 border-t border-gray-300 bg-white sticky bottom-0">
                <input
                    type="text"
                    className="flex-grow p-2 border border-gray-300 rounded-lg mr-2"
                    placeholder="Type a message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyPress}
                    disabled={loading}
                />
                <button
                    className="bg-kutcasgreen100 text-black px-4 py-2 rounded-lg hover:text-white hover:bg-kutcasgreen700"
                    onClick={sendMessage}
                    disabled={loading}
                >
                    {loading ? (
                        'Sending...'
                    ) : (
                        'Send'
                    )}
                </button>
            </div>
        </div>
    );
}

export default ChatPage;

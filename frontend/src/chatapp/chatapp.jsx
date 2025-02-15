import { useState, useEffect, useRef } from 'react';
import Navbar from '../components/navbar';
import { Icon } from "@iconify/react";

const answerShow = `
รอบการคัดเลือก: 1/1
โครงการ: นานาชาติและภาษาอังกฤษ
สาขาวิชา: วศ.บ. สาขาวิชาวิศวกรรมซอฟต์แวร์และความรู้ (นานาชาติ)
จำนวนรับ: 36
เงื่อนไขขั้นต่ำ: 1. กำลังศึกษาหรือสำเร็จการศึกษาชั้นมัธยมศึกษาปีที่ 6 หรือกำลังศึกษาในชั้นปีสุดท้ายหรือสำเร็จการศึกษาระดับมัธยมศึกษาตอนปลายจากต่างประเทศ หรือมีการเทียบวุฒิการศึกษาแบบ GED หรือเทียบเท่า โดยผู้สมัครมีผลสอบ GED ตั้งแต่เดือนพฤษภาคม 2560 เป็นต้นไป ต้องมีผลสอบ GED รวม 4 รายวิชา แต่ละวิชาต้องได้คะแนนอย่างน้อย 145 คะแนน 
2. ผลการเรียนเฉลี่ยสะสม (GPAX) 4 ภาคเรียน ไม่ต่ำกว่า 2.50 หรือเทียบเท่า  
3. ผลคะแนนสอบ  ข้อใดข้อหนึ่ง
   3.1 ผลสอบ SAT Mathematics ไม่ต่ำกว่า 600 คะแนน และ Evidence-Based Reading & Writing รวมกับ Mathematics ไม่ต่ำกว่า 1,000 คะแนน และ คะแนนสอบมาตรฐานรายวิชาภาษาอังกฤษข้อใดข้อหนึ่ง
   - ผลสอบ TOEFL (IBT) ไม่น้อยกว่า 61 คะแนน หรือเทียบเท่า
   - ผลสอบ IELTS ไม่น้อยกว่า 5.5 คะแนน หรือเทียบเท่า
   - ผลสอบ Duolingo ไม่น้อยกว่า 95 คะแนน หรือเทียบเท่า 
   3.2 ผลการเรียนเฉลี่ยสะสมรายวิชาภาษาอังกฤษ ฟิสิกส์ และคณิตศาสตร์ ในระดับชั้นมัธยมศึกษาปีที่ 4 และ 5 แต่ละวิชาไม่ต่ำกว่า 2.50 จากคะแนนเต็ม 4.00 หรือเทียบเท่า 
   3.3 สำหรับผู้สมัครที่กำลังศึกษาในปีสุดท้าย หรือสำเร็จการศึกษาระดับมัธยมศึกษาตอนปลายจากต่างประเทศ หรือมีการเทียบวุฒิการศึกษาแบบ GED หรือเทียบเท่าให้ส่งเอกสารที่แสดงว่ากำลังศึกษาในปีสุดท้ายหรือสำเร็จการศึกษา 
4. ประวัติผลงาน (Portfolio) ความยาวไม่เกิน 10 หน้ากระดาษ A4 (ไม่รวมปก คำนำ สารบัญ) รวม 1 ไฟล์ 
กำหนดการสัมภาษณ์ (ถ้าผ่านการคัดเลือก): 3 ธ.ค. 67
เกณฑ์การพิจารณา: 1. ประวัติผลงาน (Portfolio) (ควรมีผลงานตรงกับสาขาที่ต้องการสมัคร เช่น สาขาซอฟต์แวร์ ควรมีผลงานด้าน programming ฯลฯ)
2. ผลคะแนนภาษาอังกฤษ หรือระดับความสามารถในการใช้ภาษาอังกฤษ
3. การสอบสัมภาษณ์เป็นภาษาอังกฤษ
   3.1 คำถามเชิงวิชาการ/การใช้ภาษาอังกฤษ
   3.2 ทัศนคติและความเหมาะสมในการศึกษา

จากผลลัพธ์ที่ท่านต้องการหา เกณฑ์การรับเข้าศึกษาสาขาวิศวกรรมซอฟต์แวร์และความรู้ (นานาชาติ) รอบ 1/1 นานาชาติ
ผู้ใช้สามารถตรวจสอบความถูกต้องได้ที่ [อ้างอิง](https://admission.ku.ac.th/majors/project/3/)
หรือหากคำตอบไม่ตรงกับที่ท่าต้องการ ให้ลองถามด้วยรูปแบบ สาขาวิชา รอบการคัดเลือก โครงการในการเข้า และภาค เช่น วิศวะซอฟต์แวร์และความรู้ รอบ1/1 นานาชาติ ภาคนานาชาติ มีเกณฑ์อะไรบ้าง
`

function ChatPage() {
    const [messages, setMessages] = useState([
        { text: 'Hello! How can I help you?', type: 'received' },
        { text: 'Hi, I need some information.', type: 'sent' },
        { text: answerShow, type: 'received' },

    ]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState('');
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
            <Navbar />
            <div className='mx-8 mt-4 py-4 px-8 border border-gray-300 rounded-lg rounded-b-none'>
                <span className='flex items-center space-x-2'>
                    <Icon icon="fluent:bot-20-filled" width="32" height="32" style={{ color: "#0097B2", background:"#D9EBEE" , borderRadius:"50%"}} />
                    <p className='px-1 font-medium'>น้องคอนเนค</p>
                </span>
            </div>
            
            <div className="flex-grow mx-8 mb-4 p-4 overflow-y-auto border border-gray-300 rounded-lg shadow-lg rounded-t-none">
                {messages.map((msg, index) => (
                    <div key={index} className={`flex items-center mb-3 ${msg.type === 'sent' ? 'justify-end' : 'justify-start'}`}>
                        {msg.type === 'received' && (
                            <div className="w-8 h-8 bg-gray-400 rounded-full flex items-center justify-center mr-2 self-start">
                                <Icon icon="fluent:bot-20-filled" width="32" height="32" style={{ color: "#0097B2", background:"#D9EBEE" , borderRadius:"50%"}} />
                            </div>
                        )}
                        <div
                            className={`p-2 rounded-large max-w-xs ${msg.type === 'sent' ? 'bg-kutcasgreen100 text-black rounded-lg rounded-br-none' : 'bg-gray-300 bg-kutcasgreen100 text-black rounded-t-lg rounded-r-lg rounded-bl-none'}`}
                        >
                            {msg.text}
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
                />
                <button
                    className="bg-kutcasgreen100 text-black px-4 py-2 rounded-lg hover:text-white hover:bg-kutcasgreen700"
                    onClick={sendMessage}
                >
                    Send
                </button>
            </div>
        </div>
    );
}

export default ChatPage;

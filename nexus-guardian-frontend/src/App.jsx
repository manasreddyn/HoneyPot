import React, { useState, useRef, useEffect } from 'react';
import { Shield, Activity, Database, Brain, Terminal, MessageSquare, Send, AlertTriangle, Lock, Eye, Zap, ChevronRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

// Constants
const API_URL = 'http://localhost:8000/api/analyze';

// 4. Signal Fusion Engine - Mapped Signals
const getSignals = (analysis, messageText = "") => {
  if (!analysis) return [];
  const scores = analysis.model_scores || {};
  const txt = messageText.toLowerCase();

  // Keyword overrides for immediate high risk
  const hasRemoteAccess = txt.includes('teamviewer') || txt.includes('anydesk') || txt.includes('support') || txt.includes('app');
  const hasUrgency = txt.includes('immediate') || txt.includes('block') || txt.includes('suspend') || txt.includes('freeze');
  const hasBanking = txt.includes('bank') || txt.includes('sbi') || txt.includes('hdfc') || txt.includes('otp');

  // Helper to determine textual level with overrides
  const getLevel = (score, override = false) => {
    if (override) return "High";
    if (score >= 0.8) return "High";
    if (score >= 0.5) return "Medium";
    return "Low";
  };

  // derived risk signals
  const score_urgent = Math.max(scores.transformer || 0, scores.lstm || 0);
  const score_intent = Math.max(scores.pattern || 0, scores.cnn_rnn || 0);
  const score_behavior = scores.behavioral || 0;

  // Metadata is only risky if content is risky
  const contentIsRisky = score_urgent > 0.4 || score_intent > 0.4 || score_behavior > 0.4 || hasUrgency || hasBanking || hasRemoteAccess;

  return [
    {
      label: "Linguistic Manipulation",
      sub: "Urgency and authority cues",
      score: hasUrgency ? 0.95 : score_urgent,
      level: getLevel(score_urgent, hasUrgency)
    },
    {
      label: "Semantic Intent",
      sub: "Account suspension narrative",
      score: hasBanking ? 0.9 : score_intent,
      level: getLevel(score_intent, hasBanking)
    },
    {
      label: "Behavioral Pressure",
      sub: "Time-bound threat escalation",
      score: (hasUrgency && hasBanking) ? 0.9 : score_behavior,
      level: getLevel(score_behavior, hasUrgency && hasBanking)
    },
    {
      label: "External Redirection",
      sub: "Suspicious link or contact",
      score: hasRemoteAccess ? 0.95 : (scores.graph_score || 0.4),
      level: getLevel(scores.graph_score || 0.4, hasRemoteAccess)
    },
    {
      label: "Contextual Metadata",
      sub: "Channel SMS, Locale IN",
      score: contentIsRisky ? 0.95 : 0.1, // Downgrade if content is clean
      level: contentIsRisky ? "High" : "Low"
    }
  ];
};

function App() {
  const [messages, setMessages] = useState([
    // Initial system status message is distinct from the chat flow
  ]);
  const [inputText, setInputText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [latestReply, setLatestReply] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const scrollRef = useRef(null);
  const intelRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (intelRef.current) {
      intelRef.current.scrollTop = intelRef.current.scrollHeight;
    }
  }, [analysis]);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const newMessage = { id: Date.now(), sender: 'user', text: inputText };
    setMessages(prev => [...prev, newMessage]);
    setInputText('');
    setIsAnalyzing(true);
    setLatestReply(null);

    try {
      const payload = {
        sessionId: "SESSION-ACTIVE",
        message: { text: newMessage.text, timestamp: Date.now() },
        conversationHistory: messages.map(m => ({ text: m.text, sender: m.sender === 'user' ? 'scammer' : 'system' })),
        metadata: { channel: 'WEB' }
      };

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      const txt = newMessage.text.toLowerCase();

      // 1. Critical Threat Vectors (Overrides for immediate visualization)
      const indicators = {
        remoteAccess: /teamviewer|anydesk|support app|quicksupport|screen share|apk|install/.test(txt),
        bankingThreat: /(block|suspend|freeze|close).*(account|card|bank|sbi|hdfc|icici)|(kyc|pan|aadhaar).*(update|expire|pending)|(stop).*(transaction|payment)/.test(txt),
        legalThreat: /police|arrest|cbi|fir|case|court|seized|illegal|drugs|customs/.test(txt),
        financialFraud: /pay|transfer|upi|amount|rs\.|inr|refund| cashback/.test(txt) && /link|click|pin|merchant/.test(txt)
      };

      const hasScamIntent = Object.values(indicators).some(v => v);

      // 2. Risk Gating (Simply boost score if intent is found, no complex silencing)
      if (hasScamIntent && data.analysis) {
        data.analysis.risk_level = "CRITICAL";
        data.analysis.score = Math.max(data.analysis.score, 0.95);
      }

      setAnalysis(data.analysis);

      // 3. Normalized Engagement Rule
      // Engage if:
      // A) Backend says active (default logic)
      // B) OR Scammer intent is explicitly detected (Safety Net)
      // C) UNLESS message is clearly empty/short noise

      const shouldEngage = (data.honeypot && data.honeypot.active) || hasScamIntent;

      if (shouldEngage && txt.length > 2) {
        // Ensure active frame
        if (data.honeypot) data.honeypot.active = true;

        // Fallback reply if backend missed it but frontend caught intent
        const fallbackReply = "I am a bit confused. Can you explain why I need to do this?";
        const replyText = (data.honeypot && data.honeypot.reply) ? data.honeypot.reply : fallbackReply;

        setTimeout(() => {
          setLatestReply({
            id: Date.now() + 1,
            sender: 'honeypot',
            text: replyText,
            isHoneypot: true
          });
        }, 1500);
      } else {
        // Passive Mode
        if (data.honeypot) data.honeypot.active = false;
        setLatestReply(null);
      }

    } catch (error) {
      console.error("Analysis Failed", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const currentMessageText = messages.length > 0 ? messages[messages.length - 1].text : "";
  const signals = getSignals(analysis, currentMessageText);
  const isCritical = analysis?.risk_level === 'CRITICAL' || analysis?.score > 0.8;

  return (
    <div className="min-h-screen bg-[#050507] text-[#e0e0e0] font-sans p-4 md:p-6 flex flex-col gap-6">

      {/* 1. Threat Assessment Header */}
      <header className="bg-[#0a0a0f] border border-[#1a1a24] rounded-xl p-6 flex flex-col md:flex-row justify-between items-start md:items-center shadow-lg relative overflow-hidden gap-4 sticky top-0 z-50">
        {isCritical && <div className="absolute top-0 right-0 w-1/3 h-full bg-red-500/5 blur-3xl pointer-events-none"></div>}

        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-1">
            <Activity className={clsx("w-5 h-5", isCritical ? "text-red-500" : "text-emerald-500")} />
            <h1 className="text-xl font-medium tracking-wide text-white">Threat Assessment</h1>
          </div>
          <p className="text-sm text-gray-500 font-mono">
            {isCritical ? "Multiple high-risk signals aligned across channels" : "System remains in passive monitoring mode"}
          </p>
        </div>

        <div className="z-10 w-full md:w-auto flex justify-center md:justify-end">
          <div className={clsx(
            "flex items-center gap-5 px-6 py-4 rounded-lg border transition-all duration-500 bg-[#0d0d14]",
            isCritical
              ? "border-red-500/40 shadow-[0_0_25px_rgba(220,38,38,0.1)]"
              : analysis ? "border-orange-500/30" : "border-emerald-500/30"
          )}>
            {/* Risk Label */}
            <div className={clsx(
              "text-3xl md:text-4xl font-bold tracking-widest uppercase leading-none",
              isCritical ? "text-red-500" : analysis ? "text-orange-500" : "text-emerald-500"
            )}>
              {analysis?.risk_level || "MONITORING"}
            </div>

            {/* Percentage */}
            {analysis && (
              <div className="flex flex-col items-start border-l border-white/10 pl-5 py-0.5 justify-center">
                <span className="text-[10px] text-white/30 font-mono uppercase tracking-widest leading-none mb-1.5">Estimated Likelihood</span>
                <span className="text-2xl md:text-3xl font-mono text-white/50 leading-none font-light">
                  {(Math.min(analysis.score * 100, 100)).toFixed(1)}%
                </span>
              </div>
            )}
            {!analysis && (
              <div className="text-xs text-center text-white/30 font-mono pl-4 border-l border-white/10">WAITING FOR<br />INPUT</div>
            )}
          </div>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">

        {/* 2. Incoming Message Panel (Left Col) */}
        <div className="lg:col-span-4 flex flex-col gap-6 h-[70vh] lg:h-[calc(100vh-200px)] min-h-[500px]">
          <div className="bg-[#0a0a0f] border border-[#1a1a24] rounded-xl flex-1 flex flex-col overflow-hidden shadow-lg relative">
            <div className="p-4 border-b border-[#1a1a24] bg-[#0d0d14] flex justify-between items-center shrink-0 z-10">
              <div>
                <h2 className="text-sm font-semibold text-[#00f0ff] tracking-wide">Incoming Message</h2>
                <p className="text-[10px] text-gray-500 uppercase tracking-widest mt-0.5">Secure Monitoring Channel</p>
              </div>
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar relative" ref={scrollRef}>

              {/* Latest Incoming Message (API: message) */}
              <div className="p-4 border-b border-[#1a1a24] bg-[#0d0d14]/50">
                <div className="text-[10px] text-gray-500 font-mono mb-2 flex justify-between items-center">
                  <span className="uppercase tracking-widest">Latest Incoming Message (API: message)</span>
                  <span className="text-emerald-500/50 text-[9px]">Live Stream</span>
                </div>
                {messages.filter(m => m.sender === 'user').slice(-1).map((msg) => (
                  <div key={msg.id} className="p-4 rounded-lg text-sm leading-relaxed w-full bg-[#13131f] border-l-2 border-[#00f0ff] text-gray-200 font-mono shadow-lg">
                    {msg.text}
                  </div>
                ))}
                {messages.filter(m => m.sender === 'user').length === 0 && (
                  <div className="text-xs text-gray-700 italic py-2">Waiting for new input...</div>
                )}
              </div>

              {/* Conversation History (API: conversationHistory) */}
              <div className="p-4 space-y-4">
                <div className="flex flex-col gap-1 mb-2">
                  <div className="text-[10px] text-gray-500 font-mono uppercase tracking-widest">Conversation History (API: conversationHistory)</div>
                  <div className="text-[9px] text-gray-600 font-mono">Previous messages in this session. Empty for first message.</div>
                </div>

                {/* System init message is visually part of history context */}
                <div className="text-[10px] font-mono text-gray-600 text-center py-2 border-y border-dashed border-[#1a1a24] bg-[#1a1a24]/20">
                  NEXUS-GUARDIAN initialized. Monitoring conversational risk signals...
                </div>

                {messages.slice(0, Math.max(0, messages.length - 1)).reverse().map((msg) => (
                  <div key={msg.id} className={clsx("flex flex-col gap-1 max-w-[90%]", msg.sender === 'user' ? "items-start self-start" : "items-end self-end ml-auto")}>
                    <span className="text-[9px] uppercase font-mono tracking-wider text-gray-600 px-1">
                      {msg.sender === 'user' ? 'Previous Input' : 'System Reply'}
                    </span>
                    <div className={clsx(
                      "px-3 py-2 rounded text-xs leading-relaxed font-mono border-l-2",
                      msg.sender === 'user'
                        ? "bg-[#13131f]/50 border-gray-700 text-gray-400"
                        : "bg-[#1c1c2e]/50 border-emerald-900 text-emerald-700"
                    )}>
                      {msg.text}
                    </div>
                  </div>
                ))}

                {messages.length <= 1 && (
                  <div className="text-center text-gray-800 font-mono text-[10px] py-4 italic">
                    No history available.
                  </div>
                )}
              </div>
            </div>

            {/* Scroll indicator overlay */}
            <div className="absolute bottom-[80px] left-0 right-0 h-8 scroll-fade-bottom pointer-events-none"></div>

            <div className="p-4 border-t border-[#1a1a24] bg-[#0d0d14] shrink-0 z-10">
              <div className="relative">
                <textarea
                  rows={1}
                  value={inputText}
                  onChange={(e) => {
                    setInputText(e.target.value);
                    e.target.style.height = 'auto';
                    e.target.style.height = e.target.scrollHeight + 'px';
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                      setTimeout(() => e.target.style.height = 'auto', 0);
                    }
                  }}
                  placeholder="Inject mock suspect payload..."
                  className="w-full bg-[#050507] border border-[#1a1a24] rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-[#00f0ff] text-gray-300 placeholder-gray-700 resize-none overflow-hidden min-h-[46px] max-h-[150px] font-mono"
                  style={{ height: 'auto' }}
                />
                <button
                  onClick={handleSendMessage}
                  className="absolute right-2 bottom-2 p-1.5 rounded-md text-[#00f0ff] hover:bg-[#00f0ff]/10 transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* 4. Signal Fusion & 5. Cognitive Core (Center Col) */}
        <div className="lg:col-span-4 flex flex-col gap-6">

          {/* Signal Fusion Engine */}
          <div className="bg-[#0a0a0f] border border-[#1a1a24] rounded-xl overflow-hidden shadow-lg flex-1 min-h-[300px] flex flex-col">
            <div className="p-4 border-b border-[#1a1a24] bg-[#0d0d14] flex justify-between shrink-0">
              <h2 className="text-sm font-semibold text-[#00f0ff] tracking-wide">Signal Fusion Engine</h2>
              <Brain className="w-4 h-4 text-[#00f0ff]" />
            </div>

            <div className="p-4 flex flex-col gap-4 overflow-y-auto custom-scrollbar flex-1 relative">
              {signals.length > 0 ? signals.map((signal, i) => (
                <div key={i} className="group">
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="text-xs font-medium text-gray-300">{signal.label}</span>
                    <span className={clsx("text-xs font-mono font-bold",
                      signal.level.includes("High") ? "text-orange-400" : "text-gray-500"
                    )}>{signal.level}</span>
                  </div>
                  <div className="h-1 bg-[#1a1a24] rounded-full overflow-hidden mb-1">
                    <div
                      className={clsx("h-full rounded-full transition-all duration-1000",
                        signal.level === "High" ? "bg-orange-500" : "bg-[#00f0ff]"
                      )}
                      style={{ width: `${Math.min(signal.score * 100, 100)}%` }}
                    ></div>
                  </div>
                  <div className="text-[10px] text-gray-600 font-mono">{signal.sub}</div>
                </div>
              )) : (
                <div className="text-sm text-gray-600 text-center py-10 italic">Waiting for signal input...</div>
              )}
            </div>
          </div>

          {/* Cognitive Core */}
          <div className="bg-[#0a0a0f] border border-[#1a1a24] rounded-xl overflow-hidden shadow-lg min-h-[250px] flex flex-col">
            <div className="p-4 border-b border-[#1a1a24] bg-[#0d0d14] flex justify-between shrink-0">
              <h2 className="text-sm font-semibold text-[#00f0ff] tracking-wide">Cognitive Core</h2>
              <Terminal className="w-4 h-4 text-[#00f0ff]" />
            </div>
            <div className="p-4 font-mono text-xs space-y-4 overflow-y-auto custom-scrollbar flex-1">
              <div className="flex gap-3">
                <ChevronRight className="w-3 h-3 text-gray-600 mt-0.5" />
                <div>
                  <div className="text-gray-400 mb-0.5">Supervisor Agent</div>
                  <div className={clsx(isCritical ? "text-orange-400" : "text-gray-500")}>
                    {isCritical ? "Assessing threat context" : "Idle"}
                  </div>
                </div>
              </div>
              <div className="flex gap-3">
                <ChevronRight className="w-3 h-3 text-gray-600 mt-0.5" />
                <div>
                  <div className="text-gray-400 mb-0.5">Decision Engine</div>
                  <div className="text-[#00f0ff]">
                    {analysis ? "Engage for intelligence extraction" : "Standby"}
                  </div>
                </div>
              </div>
              <div className="flex gap-3">
                <ChevronRight className="w-3 h-3 text-gray-600 mt-0.5" />
                <div>
                  <div className="text-gray-400 mb-0.5">Persona Module</div>
                  <div className="text-emerald-500">
                    {analysis ? "Simulating concerned end-user" : "Inactive"}
                  </div>
                </div>
              </div>
              {isCritical && (
                <div className="flex gap-3 mt-4 pt-3 border-t border-[#1a1a24]">
                  <AlertTriangle className="w-3 h-3 text-red-500 mt-0.5" />
                  <div className="text-red-500 font-bold">Status: Scam behavior confirmed</div>
                </div>
              )}
            </div>
          </div>

        </div>

        {/* Right Col: Output & Intel */}
        <div className="lg:col-span-4 flex flex-col gap-6">

          {/* 6. Simulated User Response Panel */}
          <div className="bg-[#0a0a0f] border border-[#1a1a24] rounded-xl overflow-hidden shadow-lg p-0 flex flex-col min-h-[200px]">
            <div className="p-4 border-b border-[#1a1a24] bg-[#0d0d14] flex justify-between">
              <h2 className="text-sm font-semibold text-[#00f0ff] tracking-wide">Simulated User Response</h2>
              <MessageSquare className="w-4 h-4 text-[#00f0ff]" />
            </div>

            <div className="p-6 flex-1 flex flex-col justify-center">
              {latestReply ? (
                <div className="space-y-4">
                  <div className="font-mono text-sm text-emerald-100 bg-[#1c1c2e] border-l-2 border-emerald-500 p-4 rounded-r-lg shadow-inner">
                    "{latestReply.text}"
                  </div>
                  <div className="text-[10px] text-gray-500 font-mono text-center pt-2">
                    Objective: Increase scammer confidence and extract actionable intelligence
                  </div>
                </div>
              ) : isAnalyzing ? (
                <div className="text-xs text-gray-500 font-mono text-center italic animate-pulse">
                  Generating autonomous response...
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3 text-center px-6 opacity-60">
                  <Shield className="w-8 h-8 text-gray-600" />
                  <div className="text-sm text-gray-500 font-medium">No response generated</div>
                  <div className="text-xs text-gray-600 font-mono">
                    System in passive monitoring mode
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

// Helper: Process and deduplicate intelligence strictly
const processIntelligence = (rawIntel) => {
  if (!rawIntel) return null;

  const categories = {
    "Phone Numbers": new Set(),
    "Bank Accounts": new Set(),
    "UPI IDs": new Set(),
    "Phishing Links": new Set(),
    "Suspicious Keywords": new Set()
  };

  const seenValues = new Set();

  // Flatten and process all raw values
  Object.values(rawIntel).flat().forEach(item => {
    if (!item || typeof item !== 'string') return;
    const val = item.trim();
    if (seenValues.has(val) || val.length < 3) return; // Global dedupe & noise filter

    // Strict Classification Logic
    const cleanNum = val.replace(/[- ]/g, '');
    const isDigitOnly = /^\d+$/.test(cleanNum);

    if (/https?:\/\/|www\./.test(val)) {
      categories["Phishing Links"].add(val);
      seenValues.add(val);
    } else if (/[\w.-]+@[\w.-]+/.test(val) && !val.includes(' ')) {
      categories["UPI IDs"].add(val);
      seenValues.add(val);
    } else if (isDigitOnly) {
      // Disambiguate Phone vs Account
      if (cleanNum.length === 10 || (cleanNum.length > 10 && val.startsWith('+'))) {
        categories["Phone Numbers"].add(val);
        seenValues.add(val);
      } else if (cleanNum.length >= 9 && cleanNum.length <= 18) {
        categories["Bank Accounts"].add(val);
        seenValues.add(val);
      }
    } else {
      // Keywords - filter out timestamps or junk
      if (!val.includes(':') && val.length < 40) {
        categories["Suspicious Keywords"].add(val);
        seenValues.add(val);
      }
    }
  });

  // Filter empty categories
  return Object.entries(categories).filter(([_, set]) => set.size > 0);
};

export default App;

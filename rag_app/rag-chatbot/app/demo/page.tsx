'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import Markdown from 'react-markdown';
import Navbar from '@/components/Navbar';
import { useAuth } from '@/context/AuthContext';
import { apiUrl, fetchJson } from '@/lib/ragApi';
import styles from './page.module.css';

type DocumentMeta = {
    id: string;
    name: string;
    pages: number;
    chunks: number;
    table_chunks: number;
    image_chunks: number;
    status: string;
    stage: string;
    error?: string | null;
    backend: string;
    model: string;
    created_at: number;
};

type InfoMeta = {
    llm_backend: string;
    llm_model: string;
    embed_backend: string;
    rerank_enabled: boolean;
};

type SuggestionPayload = {
    questions: string[];
};

type ChatHit = {
    citation: string;
    chunk_id: number;
    kind: string;
    text: string;
    dense_score: number;
    sparse_score: number;
    score: number;
    rerank_score: number;
    image_file?: string | null;
};

type ChatUsage = {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
};

type Message = {
    id: string;
    role: 'user' | 'assistant';
    question?: string;
    content: string;
    loading?: boolean;
    citations?: ChatHit[];
    rewritten?: string;
    refused?: boolean;
    usage?: ChatUsage;
    error?: string;
    feedback?: 'up' | 'down' | null;
};

type ChatSession = {
    id: string;
    title: string;
    createdAt: number;
    updatedAt: number;
    messages: Message[];
};

type ChatSessionMap = Record<string, ChatSession[]>;

const DEFAULT_SUGGESTIONS = [
    'What are the key highlights in this document?',
    'Summarize the main points with citations.',
    'Which numbers or dates matter most here?',
    'What does the document say about the latest results?',
];

const ACTIVE_DOC_HINT = 'Use the selector below to switch documents or upload a new PDF.';
const CHAT_HISTORY_STORAGE_KEY = 'rag-chat-history-v1';

function formatNumber(value: number) {
    return new Intl.NumberFormat('en-US').format(value);
}

function formatRelativeTime(timestamp: number) {
    const delta = Date.now() - timestamp * 1000;
    const minutes = Math.max(1, Math.round(delta / 60000));
    if (minutes < 60) {
        return `${minutes}m ago`;
    }
    const hours = Math.round(minutes / 60);
    if (hours < 24) {
        return `${hours}h ago`;
    }
    const days = Math.round(hours / 24);
    return `${days}d ago`;
}

function getStatusClass(status: string) {
    if (status === 'ready') {
        return styles.statusReady;
    }

    if (status === 'error') {
        return styles.statusError;
    }

    if (status === 'pending' || status === 'indexing' || status === 'embedding') {
        return styles.statusPending;
    }

    return styles.statusDefault;
}

function safeLocalHistory(messages: Message[]) {
    return messages
        .filter((message) => message.role === 'user' || (message.role === 'assistant' && !message.loading))
        .map((message) => ({
            role: message.role,
            content: message.content,
        }));
}

function createMessageId() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function extractAnswerSection(content: string) {
    const splitMatch = content.match(/\n##\s+Supporting Evidence\b/i);
    if (splitMatch && splitMatch.index !== undefined) {
        return content.slice(0, splitMatch.index).trim();
    }

    const fallbackMatch = content.match(/\nSupporting Evidence\b/i);
    if (fallbackMatch && fallbackMatch.index !== undefined) {
        return content.slice(0, fallbackMatch.index).trim();
    }

    return content.trim();
}

function formatSessionTime(timestamp: number) {
    return new Intl.DateTimeFormat('en-US', {
        hour: 'numeric',
        minute: '2-digit',
    }).format(new Date(timestamp));
}

function summarizeChatTitle(messages: Message[], fallbackIndex: number) {
    const firstUserMessage = messages.find((message) => message.role === 'user');
    if (!firstUserMessage?.content) {
        return `Chat ${fallbackIndex + 1}`;
    }

    const compact = firstUserMessage.content.replace(/\s+/g, ' ').trim();
    return compact.length > 34 ? `${compact.slice(0, 31)}...` : compact;
}

function cloneChatMessages(messages: Message[]) {
    return messages.map((message) => ({ ...message }));
}

function readChatSessions() {
    if (typeof window === 'undefined') {
        return {} as ChatSessionMap;
    }

    try {
        const raw = window.localStorage.getItem(CHAT_HISTORY_STORAGE_KEY);
        return raw ? (JSON.parse(raw) as ChatSessionMap) : {};
    } catch {
        return {} as ChatSessionMap;
    }
}

function writeChatSessions(sessions: ChatSessionMap) {
    if (typeof window === 'undefined') {
        return;
    }

    window.localStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(sessions));
}

export default function DemoPage() {
    const router = useRouter();
    const { currentUser, ready } = useAuth();
    const [documents, setDocuments] = useState<DocumentMeta[]>([]);
    const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
    const [info, setInfo] = useState<InfoMeta | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [suggestions, setSuggestions] = useState<string[]>(DEFAULT_SUGGESTIONS);
    const [draft, setDraft] = useState('');
    const [loading, setLoading] = useState(true);
    const [uploading, setUploading] = useState(false);
    const [sending, setSending] = useState(false);
    const [error, setError] = useState('');
    const [feedbackError, setFeedbackError] = useState('');
    const [statusMessage, setStatusMessage] = useState('');
    const [selectedCitation, setSelectedCitation] = useState<ChatHit | null>(null);
    const [leftCollapsed, setLeftCollapsed] = useState(false);
    const [rightCollapsed, setRightCollapsed] = useState(false);
    const [chatSessionsByDoc, setChatSessionsByDoc] = useState<ChatSessionMap>({});
    const [activeChatId, setActiveChatId] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const messagesEndRef = useRef<HTMLDivElement | null>(null);

    const selectedDoc = useMemo(
        () => documents.find((document) => document.id === selectedDocId) ?? null,
        [documents, selectedDocId],
    );

    const selectedDocSessions = useMemo(
        () => (selectedDocId ? chatSessionsByDoc[selectedDocId] ?? [] : []),
        [chatSessionsByDoc, selectedDocId],
    );

    const activeChatSession = useMemo(
        () => selectedDocSessions.find((session) => session.id === activeChatId) ?? null,
        [activeChatId, selectedDocSessions],
    );

    const persistCurrentChat = (docId: string, sessionId: string, nextMessages: Message[]) => {
        if (!nextMessages.some((message) => message.role === 'user' || (message.role === 'assistant' && !message.loading))) {
            return;
        }

        setChatSessionsByDoc((currentSessions) => {
            const docSessions = currentSessions[docId] ?? [];
            const existingSession = docSessions.find((session) => session.id === sessionId);
            const updatedSession: ChatSession = {
                id: sessionId,
                title: existingSession?.title ?? summarizeChatTitle(nextMessages, docSessions.length),
                createdAt: existingSession?.createdAt ?? Date.now(),
                updatedAt: Date.now(),
                messages: cloneChatMessages(nextMessages),
            };

            const nextDocSessions = existingSession
                ? docSessions.map((session) => (session.id === sessionId ? updatedSession : session))
                : [updatedSession, ...docSessions];

            return {
                ...currentSessions,
                [docId]: nextDocSessions,
            };
        });
    };

    const loadChatSession = (sessionId: string) => {
        if (!selectedDocId) {
            return;
        }

        const session = selectedDocSessions.find((entry) => entry.id === sessionId);
        if (!session) {
            return;
        }

        setActiveChatId(session.id);
        setMessages(cloneChatMessages(session.messages));
        setDraft('');
        setError('');
        setFeedbackError('');
        setStatusMessage(`Loaded ${session.title}.`);
    };

    useEffect(() => {
        if (!ready) {
            return;
        }

        if (!currentUser) {
            router.replace('/login?returnTo=/demo');
        }
    }, [currentUser, ready, router]);

    useEffect(() => {
        if (!ready || !currentUser) {
            return;
        }

        setChatSessionsByDoc(readChatSessions());
    }, [currentUser, ready]);

    useEffect(() => {
        if (!ready || !currentUser) {
            return;
        }

        writeChatSessions(chatSessionsByDoc);
    }, [chatSessionsByDoc, currentUser, ready]);

    useEffect(() => {
        if (!ready || !currentUser) {
            return;
        }

        let cancelled = false;

        const loadInitialData = async () => {
            try {
                setLoading(true);
                const [infoPayload, docsPayload] = await Promise.all([
                    fetchJson<InfoMeta>('/api/info'),
                    fetchJson<DocumentMeta[]>('/api/documents'),
                ]);

                if (cancelled) {
                    return;
                }

                setInfo(infoPayload);
                setDocuments(docsPayload);

                if (!selectedDocId && docsPayload.length > 0) {
                    const preferredDoc = docsPayload.find((document) => document.status === 'ready') ?? docsPayload[0];
                    setSelectedDocId(preferredDoc.id);
                    setMessages([]);
                    setDraft('');
                    setSuggestions(DEFAULT_SUGGESTIONS);
                }
            } catch (loadError) {
                if (!cancelled) {
                    setError(loadError instanceof Error ? loadError.message : 'Unable to load the demo workspace.');
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        };

        void loadInitialData();

        return () => {
            cancelled = true;
        };
    }, [currentUser, ready, selectedDocId]);

    useEffect(() => {
        if (!selectedDocId) {
            return;
        }

        const selected = selectedDoc;
        if (!selected || selected.status === 'ready') {
            return;
        }

        let cancelled = false;

        const refreshDoc = async () => {
            try {
                const docPayload = await fetchJson<DocumentMeta>(`/api/documents/${selectedDocId}`);
                if (cancelled) {
                    return;
                }

                setDocuments((currentDocuments) =>
                    currentDocuments.map((document) => (document.id === docPayload.id ? docPayload : document)),
                );

                if (docPayload.status === 'ready') {
                    setStatusMessage(`${docPayload.name} is ready.`);
                    try {
                        const suggestionPayload = await fetchJson<SuggestionPayload>(`/api/documents/${docPayload.id}/suggestions`);
                        if (!cancelled) {
                            setSuggestions(suggestionPayload.questions.length > 0 ? suggestionPayload.questions : DEFAULT_SUGGESTIONS);
                        }
                    } catch {
                        if (!cancelled) {
                            setSuggestions(DEFAULT_SUGGESTIONS);
                        }
                    }
                }
            } catch {
                // Ignore transient polling failures.
            }
        };

        void refreshDoc();
        const intervalId = window.setInterval(refreshDoc, 2500);

        return () => {
            cancelled = true;
            window.clearInterval(intervalId);
        };
    }, [selectedDoc, selectedDocId]);

    useEffect(() => {
        if (!selectedDocId || !selectedDoc || selectedDoc.status !== 'ready') {
            return;
        }

        let cancelled = false;

        const loadSuggestions = async () => {
            try {
                const payload = await fetchJson<SuggestionPayload>(`/api/documents/${selectedDocId}/suggestions`);
                if (!cancelled) {
                    setSuggestions(payload.questions.length > 0 ? payload.questions : DEFAULT_SUGGESTIONS);
                }
            } catch {
                if (!cancelled) {
                    setSuggestions(DEFAULT_SUGGESTIONS);
                }
            }
        };

        void loadSuggestions();

        return () => {
            cancelled = true;
        };
    }, [selectedDoc, selectedDocId]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, [messages]);

    useEffect(() => {
        if (!selectedDocId || !activeChatId) {
            return;
        }

        persistCurrentChat(selectedDocId, activeChatId, messages);
    }, [activeChatId, messages, selectedDocId]);

    const selectDocument = (documentId: string) => {
        setSelectedDocId(documentId);
        setActiveChatId(null);
        setMessages([]);
        setDraft('');
        setSuggestions(DEFAULT_SUGGESTIONS);
        setError('');
        setFeedbackError('');
        setStatusMessage('');
    };

    const refreshDocuments = async () => {
        const docsPayload = await fetchJson<DocumentMeta[]>('/api/documents');
        setDocuments(docsPayload);
        return docsPayload;
    };

    const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        event.target.value = '';

        if (!file) {
            return;
        }

        setUploading(true);
        setError('');
        setFeedbackError('');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(apiUrl('/api/documents'), {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            const createdDocument = (await response.json()) as DocumentMeta;
            const nextDocuments = await refreshDocuments();
            const matchedDocument = nextDocuments.find((document) => document.id === createdDocument.id) ?? createdDocument;
            selectDocument(matchedDocument.id);
            setStatusMessage(`${matchedDocument.name} uploaded successfully.`);
        } catch (uploadError) {
            setError(uploadError instanceof Error ? uploadError.message : 'Upload failed.');
        } finally {
            setUploading(false);
        }
    };

    const removeDocument = async (documentId: string) => {
        setError('');

        try {
            const response = await fetch(apiUrl(`/api/documents/${documentId}`), {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            setChatSessionsByDoc((currentSessions) => {
                if (!currentSessions[documentId]) {
                    return currentSessions;
                }

                const nextSessions = { ...currentSessions };
                delete nextSessions[documentId];
                return nextSessions;
            });

            const nextDocuments = await refreshDocuments();
            if (selectedDocId === documentId) {
                const fallbackDocument = nextDocuments.find((document) => document.id !== documentId) ?? null;
                if (fallbackDocument) {
                    selectDocument(fallbackDocument.id);
                } else {
                    setSelectedDocId(null);
                    setActiveChatId(null);
                    setMessages([]);
                    setDraft('');
                    setSuggestions(DEFAULT_SUGGESTIONS);
                }
            }
        } catch (deleteError) {
            setError(deleteError instanceof Error ? deleteError.message : 'Unable to delete the document.');
        }
    };

    const clearComposer = () => {
        setDraft('');
        setStatusMessage('Composer cleared.');
    };

    const resetConversation = () => {
        setActiveChatId(null);
        setMessages([]);
        setDraft('');
        setSuggestions(DEFAULT_SUGGESTIONS);
        setStatusMessage('Started a fresh chat.');
    };

    const submitQuestion = async (questionOverride?: string) => {
        if (!selectedDoc || selectedDoc.status !== 'ready' || sending) {
            return;
        }

        const question = (questionOverride ?? draft).trim();
        if (!question) {
            return;
        }

        const sessionId = activeChatId ?? createMessageId();
        if (!activeChatId) {
            setActiveChatId(sessionId);
        }

        const userMessageId = createMessageId();
        const assistantMessageId = createMessageId();
        const history = safeLocalHistory(messages);

        setMessages((currentMessages) => [
            ...currentMessages,
            { id: userMessageId, role: 'user', content: question },
            { id: assistantMessageId, role: 'assistant', content: '', loading: true, question, feedback: null },
        ]);
        setDraft('');
        setSending(true);
        setError('');
        setFeedbackError('');
        setStatusMessage('Answer is streaming from the backend...');

        try {
            const response = await fetch(apiUrl(`/api/documents/${selectedDoc.id}/chat/stream`), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question,
                    history,
                    top_k: 8,
                }),
            });

            if (!response.ok || !response.body) {
                throw new Error(await response.text());
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = '';
            let dataLines: string[] = [];

            const flushFrame = () => {
                if (!currentEvent || dataLines.length === 0) {
                    currentEvent = '';
                    dataLines = [];
                    return;
                }

                const payload = JSON.parse(dataLines.join('\n')) as Record<string, unknown>;

                if (currentEvent === 'meta') {
                    const hits = (payload.hits as ChatHit[] | undefined) ?? [];
                    setMessages((currentMessages) =>
                        currentMessages.map((message) =>
                            message.id === assistantMessageId
                                ? {
                                    ...message,
                                    citations: hits,
                                    rewritten: typeof payload.rewritten === 'string' ? payload.rewritten : undefined,
                                    refused: Boolean(payload.refused),
                                }
                                : message,
                        ),
                    );
                }

                if (currentEvent === 'token') {
                    const token = typeof payload.t === 'string' ? payload.t : '';
                    if (token) {
                        setMessages((currentMessages) =>
                            currentMessages.map((message) =>
                                message.id === assistantMessageId
                                    ? { ...message, content: `${message.content}${token}` }
                                    : message,
                            ),
                        );
                    }
                }

                if (currentEvent === 'done') {
                    const usage = (payload.usage as ChatUsage | undefined) ?? {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    };

                    setMessages((currentMessages) =>
                        currentMessages.map((message) =>
                            message.id === assistantMessageId ? { ...message, loading: false, usage } : message,
                        ),
                    );
                    setStatusMessage('Answer complete.');
                }

                if (currentEvent === 'error') {
                    const detail = typeof payload.detail === 'string' ? payload.detail : 'Unable to stream the answer.';
                    setMessages((currentMessages) =>
                        currentMessages.map((message) =>
                            message.id === assistantMessageId
                                ? { ...message, loading: false, error: detail, content: detail }
                                : message,
                        ),
                    );
                    throw new Error(detail);
                }

                currentEvent = '';
                dataLines = [];
            };

            while (true) {
                const { done, value } = await reader.read();
                buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

                let lineBreakIndex = buffer.indexOf('\n');
                while (lineBreakIndex >= 0) {
                    const rawLine = buffer.slice(0, lineBreakIndex).replace(/\r$/, '');
                    buffer = buffer.slice(lineBreakIndex + 1);

                    if (!rawLine) {
                        flushFrame();
                    } else if (rawLine.startsWith('event: ')) {
                        currentEvent = rawLine.slice(7).trim();
                    } else if (rawLine.startsWith('data: ')) {
                        dataLines.push(rawLine.slice(6));
                    }

                    lineBreakIndex = buffer.indexOf('\n');
                }

                if (done) {
                    if (buffer.trim()) {
                        const tailLines = buffer.split(/\r?\n/);
                        for (const line of tailLines) {
                            if (!line) {
                                flushFrame();
                            } else if (line.startsWith('event: ')) {
                                currentEvent = line.slice(7).trim();
                            } else if (line.startsWith('data: ')) {
                                dataLines.push(line.slice(6));
                            }
                        }
                    }

                    flushFrame();
                    break;
                }
            }
        } catch (submitError) {
            setMessages((currentMessages) =>
                currentMessages.map((message) =>
                    message.id === assistantMessageId
                        ? {
                            ...message,
                            loading: false,
                            error: submitError instanceof Error ? submitError.message : 'Unable to answer the question.',
                            content: submitError instanceof Error ? submitError.message : 'Unable to answer the question.',
                        }
                        : message,
                ),
            );
            setError(submitError instanceof Error ? submitError.message : 'Unable to answer the question.');
        } finally {
            setSending(false);
        }
    };

    const submitFeedback = async (message: Message, rating: 'up' | 'down') => {
        if (!selectedDoc || !message.question) {
            return;
        }

        try {
            const response = await fetch(apiUrl('/api/feedback'), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    doc_id: selectedDoc.id,
                    question: message.question,
                    answer: message.content,
                    rating,
                    citations: message.citations?.map((citation) => citation.citation) ?? [],
                }),
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            setMessages((currentMessages) =>
                currentMessages.map((entry) => (entry.id === message.id ? { ...entry, feedback: rating } : entry)),
            );
            setFeedbackError('');
        } catch (feedbackSubmissionError) {
            setFeedbackError(
                feedbackSubmissionError instanceof Error
                    ? feedbackSubmissionError.message
                    : 'Unable to submit feedback.',
            );
        }
    };

    if (!ready || loading) {
        return (
            <>
                <Navbar />
                <main className={styles.page}>
                    <div className={styles.backdrop} />
                    <div className={styles.content}>
                        <div className={styles.hero}>
                            <div className={styles.titleBlock}>
                                <span className={styles.eyebrow}>DocChat grounded · cited · refusal-safe</span>
                                <h1 className={styles.title}>Preparing your workspace</h1>
                                <p className={styles.subtitle}>
                                    Restoring the document list, model info, and chat workspace so you can continue from the same theme as the rest of the site.
                                </p>
                            </div>
                        </div>

                        <div className={styles.card}>
                            <div className={styles.cardBody}>
                                <div className={styles.skeletonLine} />
                                <div style={{ height: 14 }} />
                                <div className={styles.skeletonLine} />
                                <div style={{ height: 14 }} />
                                <div className={`${styles.skeletonLine} ${styles.skeletonLong}`} />
                                <div style={{ height: 14 }} />
                                <div className={`${styles.skeletonLine} ${styles.skeletonMedium}`} />
                            </div>
                        </div>
                    </div>
                </main>
            </>
        );
    }

    if (!currentUser) {
        return null;
    }

    const workspaceClasses = [
        styles.workspace,
        leftCollapsed ? styles.workspaceLeftCollapsed : '',
        rightCollapsed ? styles.workspaceRightCollapsed : '',
    ]
        .filter(Boolean)
        .join(' ');

    return (
        <>
            <Navbar />
            <main className={styles.page}>
                <div className={styles.backdrop} />

                <div className={styles.content}>
                    <section className={styles.hero}>
                        <div className={styles.titleBlock}>
                            <span className={styles.eyebrow}>DocChat grounded · cited · refusal-safe</span>
                            <h1 className={styles.title}>RAG chatbot workspace</h1>
                            <p className={styles.subtitle}>
                                Upload a PDF, wait for indexing, then ask grounded questions with citations and source traces. {ACTIVE_DOC_HINT}
                            </p>
                        </div>

                        <div className={styles.topActions}>
                            <span className={`${styles.pill} ${styles.pillAccent}`}>
                                {info ? `${info.llm_backend} · ${info.llm_model}` : 'Loading model info'}
                            </span>
                            <span className={styles.pill}>
                                {info ? `${info.embed_backend} · ${info.embed_model}` : 'Loading embeddings'}
                            </span>
                            <span className={styles.pill}>
                                {info?.rerank_enabled ? `Rerank · ${info.rerank_backend}` : 'Rerank disabled'}
                            </span>
                        </div>
                    </section>

                    {error ? <div className={styles.errorBox}>{error}</div> : null}
                    {feedbackError ? <div className={styles.errorBox}>{feedbackError}</div> : null}
                    {statusMessage ? <div className={styles.pill}>{statusMessage}</div> : null}

                    <section className={workspaceClasses}>
                        {leftCollapsed ? (
                            <aside className={styles.collapsedRail}>
                                <button
                                    type="button"
                                    className={styles.panelRailButton}
                                    onClick={() => setLeftCollapsed(false)}
                                    aria-label="Show document panel"
                                >
                                    Show docs
                                </button>
                            </aside>
                        ) : (
                            <div className={styles.column}>
                                <article className={styles.card}>
                                    <div className={styles.cardHeader}>
                                        <div>
                                            <div className={styles.cardTitle}>Document intake</div>
                                        </div>
                                        <div className={styles.headerActions}>
                                            <button className={styles.uploadButton} type="button" onClick={() => fileInputRef.current?.click()}>
                                                {uploading ? 'Uploading...' : 'Upload PDF'}
                                            </button>
                                            <button
                                                type="button"
                                                className={styles.panelToggleButton}
                                                onClick={() => setLeftCollapsed(true)}
                                                aria-label="Collapse document panel"
                                            >
                                                Collapse
                                            </button>
                                        </div>
                                    </div>

                                    <div className={styles.cardBody}>
                                        <div className={styles.uploadArea}>
                                            <p className={styles.uploadHint}>
                                                Add a PDF to index it in the same backend used by the demo UI. Large files will take a moment to move from pending to ready.
                                            </p>
                                            <input
                                                ref={fileInputRef}
                                                type="file"
                                                accept="application/pdf"
                                                hidden
                                                onChange={handleUpload}
                                            />

                                            <div className={styles.statsRow}>
                                                <div className={styles.stat}>
                                                    <span className={styles.statValue}>{formatNumber(documents.length)}</span>
                                                    <span className={styles.statLabel}>Documents indexed</span>
                                                </div>
                                                <div className={styles.stat}>
                                                    <span className={styles.statValue}>{selectedDoc ? formatNumber(selectedDoc.pages) : '0'}</span>
                                                    <span className={styles.statLabel}>Pages in focus</span>
                                                </div>
                                                <div className={styles.stat}>
                                                    <span className={styles.statValue}>{selectedDoc ? formatNumber(selectedDoc.chunks) : '0'}</span>
                                                    <span className={styles.statLabel}>Chunks available</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </article>

                                <article className={styles.card}>
                                    <div className={styles.cardHeader}>
                                        <div className={styles.cardTitle}>Indexed documents</div>
                                        <span className={styles.miniChip}>{selectedDoc ? 'Active' : 'No document selected'}</span>
                                    </div>

                                    <div className={styles.cardBody}>
                                        <div className={styles.docList}>
                                            {documents.length === 0 ? (
                                                <div className={styles.emptyState}>
                                                    <strong>No PDFs uploaded yet.</strong>
                                                    <span>Upload one to unlock grounded questions, citations, and the live PDF link.</span>
                                                </div>
                                            ) : null}

                                            {documents.map((document) => (
                                                <div
                                                    key={document.id}
                                                    className={`${styles.docItem} ${selectedDocId === document.id ? styles.docItemActive : ''}`}
                                                    role="button"
                                                    tabIndex={0}
                                                    onClick={() => selectDocument(document.id)}
                                                    onKeyDown={(event) => {
                                                        if (event.key === 'Enter' || event.key === ' ') {
                                                            event.preventDefault();
                                                            selectDocument(document.id);
                                                        }
                                                    }}
                                                >
                                                    <div className={styles.docItemTop}>
                                                        <span className={styles.docName}>{document.name}</span>
                                                        <span className={`${styles.statusChip} ${getStatusClass(document.status)}`}>{document.status}</span>
                                                    </div>
                                                    <div className={styles.docMeta}>
                                                        {document.pages} pages · {document.chunks} chunks · {formatRelativeTime(document.created_at)}
                                                    </div>
                                                    <div className={styles.docMeta}>
                                                        {document.stage || document.error || 'Ready for conversation'}
                                                    </div>
                                                    <div className={styles.docActions}>
                                                        <a
                                                            className={styles.ghostButton}
                                                            href={apiUrl(`/api/documents/${document.id}/pdf`)}
                                                            target="_blank"
                                                            rel="noreferrer"
                                                            onClick={(event) => event.stopPropagation()}
                                                        >
                                                            Open PDF
                                                        </a>
                                                        <button
                                                            className={styles.dangerButton}
                                                            type="button"
                                                            onClick={(event) => {
                                                                event.stopPropagation();
                                                                void removeDocument(document.id);
                                                            }}
                                                        >
                                                            Delete
                                                        </button>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </article>

                                <article className={styles.card}>
                                    <div className={styles.cardHeader}>
                                        <div className={styles.cardTitle}>Active document</div>
                                        <span className={`${styles.statusChip} ${selectedDoc ? getStatusClass(selectedDoc.status) : styles.statusDefault}`}>
                                            {selectedDoc ? selectedDoc.status : 'none'}
                                        </span>
                                    </div>

                                    <div className={styles.cardBody}>
                                        {selectedDoc ? (
                                            <div className={styles.metaGrid}>
                                                <div className={styles.metaRow}>
                                                    <span className={styles.metaLabel}>Document</span>
                                                    <span className={styles.metaValue}>{selectedDoc.name}</span>
                                                </div>
                                                <div className={styles.metaRow}>
                                                    <span className={styles.metaLabel}>Backend</span>
                                                    <span className={styles.metaValue}>
                                                        {selectedDoc.backend || '—'} · {selectedDoc.model || '—'}
                                                    </span>
                                                </div>
                                                <div className={styles.metaRow}>
                                                    <span className={styles.metaLabel}>Counts</span>
                                                    <span className={styles.metaValue}>
                                                        {selectedDoc.pages} pages · {selectedDoc.chunks} chunks · {selectedDoc.table_chunks} tables · {selectedDoc.image_chunks} images
                                                    </span>
                                                </div>
                                                <div className={styles.metaRow}>
                                                    <span className={styles.metaLabel}>Status</span>
                                                    <span className={styles.metaValue}>{selectedDoc.stage || selectedDoc.error || 'Ready to chat'}</span>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className={styles.emptyState}>
                                                <strong>Choose a document to start.</strong>
                                                <span>Once a PDF is ready, the chat panel will stream grounded answers and show the supporting citations.</span>
                                            </div>
                                        )}
                                    </div>
                                </article>
                            </div>
                        )}

                        <div className={styles.column}>
                            <article className={`${styles.card} ${styles.chatCard}`}>
                                <div className={styles.cardHeader}>
                                    <div className={styles.chatHeader}>
                                        <div className={styles.chatHeaderTop}>
                                            <div className={styles.chatHeadline}>
                                                <div className={styles.cardTitle}>Conversation</div>
                                                <div className={styles.chatTitle}>
                                                    {selectedDoc ? selectedDoc.name : 'Pick a document to chat with'}
                                                </div>
                                                <div className={styles.chatSub}>
                                                    {selectedDoc?.status === 'ready'
                                                        ? 'Ask grounded questions. The backend will cite the chunks it uses.'
                                                        : 'Wait for the document to finish indexing before asking questions.'}
                                                </div>
                                            </div>
                                        </div>

                                        <div className={styles.chipRow}>
                                            <select
                                                className={styles.chatHistorySelect}
                                                value={activeChatSession?.id ?? ''}
                                                onChange={(event) => {
                                                    const sessionId = event.target.value;
                                                    if (!sessionId) {
                                                        setActiveChatId(null);
                                                        return;
                                                    }

                                                    loadChatSession(sessionId);
                                                }}
                                                disabled={!selectedDoc || selectedDocSessions.length === 0}
                                                aria-label="Chat history"
                                            >
                                                <option value="">Chat history</option>
                                                {selectedDocSessions.map((session) => (
                                                    <option key={session.id} value={session.id}>
                                                        {session.title} · {formatSessionTime(session.updatedAt)}
                                                    </option>
                                                ))}
                                            </select>
                                            <button className={styles.ghostButton} type="button" onClick={resetConversation} disabled={!selectedDoc}>
                                                New chat
                                            </button>
                                            <button className={styles.ghostButton} type="button" onClick={clearComposer}>
                                                Clear
                                            </button>
                                            {selectedDoc ? (
                                                <a
                                                    className={styles.ghostButton}
                                                    href={apiUrl(`/api/documents/${selectedDoc.id}/pdf`)}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                >
                                                    Open PDF
                                                </a>
                                            ) : null}
                                        </div>
                                    </div>
                                </div>

                                <div className={styles.cardBody}>
                                    <div className={styles.chatBody}>
                                        <div className={styles.questionChips}>
                                            {suggestions.map((suggestion) => (
                                                <button
                                                    key={suggestion}
                                                    type="button"
                                                    className={styles.questionChip}
                                                    onClick={() => void submitQuestion(suggestion)}
                                                    disabled={!selectedDoc || selectedDoc.status !== 'ready' || sending}
                                                >
                                                    {suggestion}
                                                </button>
                                            ))}
                                        </div>

                                        <div className={styles.messageList}>
                                            {messages.length === 0 ? (
                                                <div className={styles.emptyState}>
                                                    <strong>No messages yet.</strong>
                                                    <span>Try one of the suggested questions or type your own prompt below.</span>
                                                </div>
                                            ) : null}

                                            {messages.map((message) => (
                                                <div
                                                    key={message.id}
                                                    className={`${styles.message} ${message.role === 'user' ? styles.messageUser : ''}`}
                                                >
                                                    <div className={styles.messageBubble}>
                                                        {message.loading && message.content.length === 0 ? (
                                                            <>
                                                                <div className={styles.skeletonLine} />
                                                                <div style={{ height: 8 }} />
                                                                <div className={`${styles.skeletonLine} ${styles.skeletonLong}`} />
                                                            </>
                                                        ) : message.role === 'assistant' ? (
                                                            <Markdown
                                                                components={{
                                                                    h1: ({ ...props }) => <h1 style={{ fontSize: '1.5em', marginTop: '0.5em', marginBottom: '0.3em', fontWeight: 700 }} {...props} />,
                                                                    h2: ({ ...props }) => <h2 style={{ fontSize: '1.3em', marginTop: '0.4em', marginBottom: '0.2em', fontWeight: 700 }} {...props} />,
                                                                    h3: ({ ...props }) => <h3 style={{ fontSize: '1.1em', marginTop: '0.3em', marginBottom: '0.2em', fontWeight: 700 }} {...props} />,
                                                                    p: ({ ...props }) => <p style={{ marginBottom: '0.8em' }} {...props} />,
                                                                    ul: ({ ...props }) => <ul style={{ marginLeft: '1.5em', marginBottom: '0.8em' }} {...props} />,
                                                                    ol: ({ ...props }) => <ol style={{ marginLeft: '1.5em', marginBottom: '0.8em' }} {...props} />,
                                                                    li: ({ ...props }) => <li style={{ marginBottom: '0.3em' }} {...props} />,
                                                                    code: ({ inline, ...props }) =>
                                                                        inline ? (
                                                                            <code style={{ background: 'rgba(255,255,255,0.08)', padding: '0.2em 0.4em', borderRadius: '4px', fontFamily: 'var(--mono)' }} {...props} />
                                                                        ) : (
                                                                            <code style={{ display: 'block', background: 'rgba(255,255,255,0.08)', padding: '0.8em', borderRadius: '6px', fontFamily: 'var(--mono)', overflowX: 'auto', marginBottom: '0.8em' }} {...props} />
                                                                        ),
                                                                    blockquote: ({ ...props }) => <blockquote style={{ borderLeft: '3px solid var(--accent)', paddingLeft: '1em', marginLeft: 0, opacity: 0.8, marginBottom: '0.8em' }} {...props} />,
                                                                    table: ({ ...props }) => <table style={{ borderCollapse: 'collapse', marginBottom: '0.8em', width: '100%' }} {...props} />,
                                                                    th: ({ ...props }) => <th style={{ border: '1px solid var(--border)', padding: '0.5em', background: 'rgba(34,197,94,0.1)' }} {...props} />,
                                                                    td: ({ ...props }) => <td style={{ border: '1px solid var(--border)', padding: '0.5em' }} {...props} />,
                                                                    a: ({ ...props }) => <a style={{ color: 'var(--accent)', textDecoration: 'underline', cursor: 'pointer' }} {...props} />,
                                                                }}
                                                            >
                                                                {extractAnswerSection(message.content)}
                                                            </Markdown>
                                                        ) : (
                                                            message.content
                                                        )}
                                                    </div>

                                                    {message.role === 'assistant' ? (
                                                        <div className={styles.assistantMeta}>
                                                            {message.rewritten ? (
                                                                <div className={styles.messageMeta}>
                                                                    <span className={`${styles.miniChip} ${styles.pillAccent}`}>Rewritten query</span>
                                                                    <span>{message.rewritten}</span>
                                                                </div>
                                                            ) : null}

                                                            {message.citations && message.citations.length > 0 ? (
                                                                <div className={styles.citationPills}>
                                                                    {message.citations.map((citation) => (
                                                                        <button
                                                                            key={`${message.id}-${citation.citation}`}
                                                                            type="button"
                                                                            className={`${styles.citationPill} ${selectedCitation?.citation === citation.citation ? styles.citationPillActive : ''}`}
                                                                            onClick={() => setSelectedCitation(citation)}
                                                                        >
                                                                            [{citation.citation}]
                                                                        </button>
                                                                    ))}
                                                                </div>
                                                            ) : null}

                                                            <div className={styles.feedbackRow}>
                                                                <button
                                                                    type="button"
                                                                    className={`${styles.feedbackButton} ${message.feedback === 'up' ? styles.feedbackButtonActive : ''}`}
                                                                    onClick={() => void submitFeedback(message, 'up')}
                                                                    disabled={!selectedDoc}
                                                                >
                                                                    Helpful
                                                                </button>
                                                                <button
                                                                    type="button"
                                                                    className={`${styles.feedbackButton} ${message.feedback === 'down' ? styles.feedbackButtonActive : ''}`}
                                                                    onClick={() => void submitFeedback(message, 'down')}
                                                                    disabled={!selectedDoc}
                                                                >
                                                                    Not helpful
                                                                </button>
                                                                {message.usage ? (
                                                                    <span className={styles.messageMeta}>
                                                                        {message.usage.total_tokens} tokens
                                                                    </span>
                                                                ) : null}
                                                            </div>
                                                        </div>
                                                    ) : null}
                                                </div>
                                            ))}
                                            <div ref={messagesEndRef} />
                                        </div>

                                        <div className={styles.composer}>
                                            <textarea
                                                className={styles.textarea}
                                                value={draft}
                                                onChange={(event) => setDraft(event.target.value)}
                                                placeholder={
                                                    selectedDoc?.status === 'ready'
                                                        ? 'Ask about the uploaded document...'
                                                        : 'Select a ready document first.'
                                                }
                                                disabled={!selectedDoc || selectedDoc.status !== 'ready' || sending}
                                                onKeyDown={(event) => {
                                                    if (event.key === 'Enter' && !event.shiftKey) {
                                                        event.preventDefault();
                                                        void submitQuestion();
                                                    }
                                                }}
                                            />

                                            <div className={styles.composerMeta}>
                                                <div className={styles.composerHints}>Enter to send · Shift+Enter newline · / to focus</div>

                                                <div className={styles.composerActions}>
                                                    <button
                                                        type="button"
                                                        className={styles.secondaryButton}
                                                        onClick={() => void submitQuestion()}
                                                        disabled={!selectedDoc || selectedDoc.status !== 'ready' || sending}
                                                    >
                                                        {sending ? 'Thinking...' : 'Send'}
                                                    </button>
                                                    <button type="button" className={styles.ghostButton} onClick={clearComposer}>
                                                        Clear draft
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </article>
                        </div>

                        {rightCollapsed ? (
                            <aside className={`${styles.collapsedRail} ${styles.collapsedRailRight}`}>
                                <button
                                    type="button"
                                    className={styles.panelRailButton}
                                    onClick={() => setRightCollapsed(false)}
                                    aria-label="Show retrieved chunk panel"
                                >
                                    Show chunk
                                </button>
                            </aside>
                        ) : (
                            <aside className={styles.sidePanel}>
                                <article className={`${styles.card} ${styles.sideCard}`}>
                                    <div className={styles.cardHeader}>
                                        <div className={styles.cardTitle}>Retrieved chunk</div>
                                        <div className={styles.headerActions}>
                                            <span className={styles.miniChip}>{selectedCitation ? 'Open' : 'Waiting'}</span>
                                            <button
                                                type="button"
                                                className={styles.panelToggleButton}
                                                onClick={() => setRightCollapsed(true)}
                                                aria-label="Collapse retrieved chunk panel"
                                            >
                                                Collapse
                                            </button>
                                        </div>
                                    </div>

                                    <div className={styles.cardBody}>
                                        {selectedCitation ? (
                                            <div className={styles.sidePanelContent}>
                                                <div className={styles.metaGrid}>
                                                    <div className={styles.metaRow}>
                                                        <span className={styles.metaLabel}>Citation</span>
                                                        <span className={styles.metaValue}>[{selectedCitation.citation}]</span>
                                                    </div>
                                                    <div className={styles.metaRow}>
                                                        <span className={styles.metaLabel}>Location</span>
                                                        <span className={styles.metaValue}>
                                                            <a
                                                                className={styles.sideLink}
                                                                href={selectedDoc ? `${apiUrl(`/api/documents/${selectedDoc.id}/pdf`)}#page=${selectedCitation.page}` : '#'}
                                                                target="_blank"
                                                                rel="noreferrer"
                                                            >
                                                                Page {selectedCitation.page}
                                                            </a>{' '}
                                                            · chunk {selectedCitation.chunk_id} · {selectedCitation.kind}
                                                        </span>
                                                    </div>
                                                    <div className={styles.metaRow}>
                                                        <span className={styles.metaLabel}>Scores</span>
                                                        <span className={styles.metaValue}>
                                                            dense {selectedCitation.dense_score.toFixed(3)} · sparse {selectedCitation.sparse_score.toFixed(3)} · rerank {selectedCitation.rerank_score.toFixed(3)}
                                                        </span>
                                                    </div>
                                                    {selectedCitation.image_file ? (
                                                        <div className={styles.metaRow}>
                                                            <span className={styles.metaLabel}>Image</span>
                                                            <a
                                                                className={styles.sideLink}
                                                                href={selectedDoc ? apiUrl(`/api/documents/${selectedDoc.id}/images/${selectedCitation.image_file}`) : '#'}
                                                                target="_blank"
                                                                rel="noreferrer"
                                                            >
                                                                Open extracted image
                                                            </a>
                                                        </div>
                                                    ) : null}
                                                </div>

                                                <div className={styles.sideEvidenceBlock}>
                                                    <span className={styles.metaLabel}>Embedding / source text</span>
                                                    <div className={styles.sideEvidenceText}>{selectedCitation.text}</div>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className={styles.emptyState}>
                                                <strong>No chunk selected.</strong>
                                                <span>Click a citation in the conversation to load evidence, location, and scores here.</span>
                                            </div>
                                        )}
                                    </div>
                                </article>
                            </aside>
                        )}
                    </section>
                </div>
            </main>
        </>
    );
}
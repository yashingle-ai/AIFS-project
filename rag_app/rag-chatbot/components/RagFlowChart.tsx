'use client';

import { useState, useRef, useCallback } from 'react';
import styles from './RagFlowChart.module.css';

const STEPS = [
  { id: 'n1',  phase: 'index', icon: '📄', title: 'Upload PDF',              sub: 'User submits document',          info: 'Step 1 — User uploads a PDF. The system accepts the file and queues it for background processing.' },
  { id: 'n2',  phase: 'index', icon: '⚙️', title: 'Background indexing',     sub: 'Extract pages, tables, images',  info: 'Step 2 — Background indexing begins: the document is split into pages, tables are detected, and images are extracted.' },
  { id: 'n3',  phase: 'index', icon: '✂️', title: 'Chunking & enrichment',   sub: 'Retrieval-ready content',        info: 'Step 3 — Chunking breaks pages into semantically coherent segments. Enrichment adds metadata like headings, section context, and bounding boxes.' },
  { id: 'n4',  phase: 'index', icon: '📝', title: 'Generate summaries',       sub: 'Page & document level',          info: 'Step 4 — Page-level and full-document summaries are generated using an LLM. These enable multi-granularity retrieval.' },
  { id: 'n5',  phase: 'index', icon: '💾', title: 'Save indexes to disk',     sub: 'Hierarchical hybrid indexes',    info: 'Step 5 — Dense vector indexes and sparse keyword indexes are combined into a hierarchical structure and persisted to disk. Document is now query-ready.' },
  { id: 'n6',  phase: 'query', icon: '💬', title: 'User question',            sub: 'Submit query',                   info: 'Step 6 — The user submits a question. The system records this as a new chat turn.' },
  { id: 'n7',  phase: 'query', icon: '🔐', title: 'Validate & load retriever',sub: 'Backend checks request',         info: 'Step 7 — The backend validates the request (auth, rate limits) and loads the document retriever from disk.' },
  { id: 'n8',  phase: 'query', icon: '✏️', title: 'Query rewrite',            sub: 'Optionally uses history',        info: 'Step 8 — The query is optionally rewritten using conversation history to resolve pronouns and implicit references.' },
  { id: 'n9',  phase: 'query', icon: '🔍', title: 'Intent → index selection', sub: 'Choose which index levels',      info: 'Step 9 — Query intent is classified to decide which index levels to search: chunk-level, page-level, or document summaries.' },
  { id: 'n10', phase: 'query', icon: '⚡', title: 'Dense + sparse retrieval', sub: 'Fused & reranked',               info: 'Step 10 — Dense (semantic) and sparse (BM25) retrieval are run in parallel. Results are score-fused and optionally reranked.' },
  { id: 'n11', phase: 'query', icon: '⚖️', title: 'Evidence strong enough?', sub: 'Confidence threshold check',     info: 'Step 11 — A confidence threshold is evaluated. If the top-k evidence passes the minimum relevance score, the LLM proceeds to answer.' },
  { id: 'n12', phase: 'yes',   icon: '🤖', title: 'LLM writes answer',        sub: 'Grounded response',              info: 'Step 12 (YES) — The LLM generates a grounded answer using only the retrieved passages as context, with inline citations.' },
  { id: 'n13', phase: 'no',    icon: '🚫', title: 'Refuse',                   sub: 'Not found in document',          info: 'Step 12 (NO) — Evidence is too weak. The backend refuses and skips the LLM entirely.' },
  { id: 'n14', phase: 'final', icon: '📦', title: 'Response returned',        sub: 'Citations · hits · tokens · refusal', info: 'Step 13 — The final response is returned: answer text (or refusal), citation list, retrieval hit count, token usage, and a refusal flag.' },
];

const INDEX_STEPS = STEPS.filter(s => s.phase === 'index');
const QUERY_STEPS = STEPS.filter(s => s.phase === 'query');

type StepState = 'idle' | 'active' | 'done';

export default function RagFlowChart() {
  const [current, setCurrent] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed]     = useState(900);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const getState = (idx: number): StepState => {
    if (idx < current) return 'done';
    if (idx === current) return 'active';
    return 'idle';
  };

  const jumpTo = useCallback((idx: number) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setPlaying(false);
    setCurrent(idx);
  }, []);

  const startAnim = useCallback(() => {
    if (playing) return;
    setPlaying(true);
    let step = current;
    const tick = () => {
      step += 1;
      if (step >= STEPS.length) { setPlaying(false); return; }
      setCurrent(step);
      timerRef.current = setTimeout(tick, speed);
    };
    tick();
  }, [playing, current, speed]);

  const resetAnim = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setPlaying(false);
    setCurrent(-1);
  }, []);

  const progress = current >= 0 ? ((current + 1) / STEPS.length) * 100 : 0;
  const infoText = current >= 0 ? STEPS[current].info : 'Press Play to walk through the full RAG pipeline, or click any step to jump to it.';

  return (
    <div className={styles.root}>
      {/* Controls */}
      <div className={styles.controls}>
        <div className={styles.btnGroup}>
          <button className={styles.btn} onClick={startAnim} disabled={playing}>
            {playing ? '⏸ Playing…' : '▶ Play'}
          </button>
          <button className={styles.btnSecondary} onClick={resetAnim}>↺ Reset</button>
        </div>
        <div className={styles.speedWrap}>
          <span>Speed</span>
          <input type="range" min={300} max={1800} step={100} value={speed}
            onChange={e => setSpeed(Number(e.target.value))} className={styles.slider} />
        </div>
      </div>

      {/* Flow */}
      <div className={styles.flow}>

        {/* Index phase */}
        <div className={styles.column}>
          <div className={styles.phaseLabel}>Indexing phase</div>
          {INDEX_STEPS.map((s, i) => {
            const globalIdx = STEPS.indexOf(s);
            return (
              <div key={s.id}>
                <FlowNode step={s} state={getState(globalIdx)} onClick={() => jumpTo(globalIdx)} />
                {i < INDEX_STEPS.length - 1 && <div className={styles.arrow} />}
              </div>
            );
          })}
        </div>

        {/* Divider */}
        <div className={styles.divider}>
          <div className={styles.divLine} />
          <span className={styles.divLabel}>query</span>
          <div className={styles.divLine} />
        </div>

        {/* Query phase */}
        <div className={styles.column}>
          <div className={styles.phaseLabel}>Query phase</div>
          {QUERY_STEPS.map((s, i) => {
            const globalIdx = STEPS.indexOf(s);
            return (
              <div key={s.id}>
                <FlowNode step={s} state={getState(globalIdx)} onClick={() => jumpTo(globalIdx)} decision={s.id === 'n11'} />
                {i < QUERY_STEPS.length - 1 && <div className={styles.arrow} />}
              </div>
            );
          })}

          {/* YES / NO split */}
          <div className={styles.split}>
            <div className={styles.splitCol}>
              <div className={styles.arrow} />
              <div className={styles.branchLabel} style={{ color: '#16a34a' }}>YES</div>
              <FlowNode step={STEPS[11]} state={getState(11)} onClick={() => jumpTo(11)} />
            </div>
            <div className={styles.splitCol}>
              <div className={styles.arrow} />
              <div className={styles.branchLabel} style={{ color: '#dc2626' }}>NO</div>
              <FlowNode step={STEPS[12]} state={getState(12)} onClick={() => jumpTo(12)} />
            </div>
          </div>

          {/* Converge arrows */}
          <div className={styles.split} style={{ gap: 8 }}>
            <div className={styles.splitCol}><div className={styles.arrow} /></div>
            <div className={styles.splitCol}><div className={styles.arrow} /></div>
          </div>

          {/* Final node */}
          <FlowNode step={STEPS[13]} state={getState(13)} onClick={() => jumpTo(13)} />
        </div>
      </div>

      {/* Progress */}
      <div className={styles.progressRow}>
        <div className={styles.progressTrack}>
          <div className={styles.progressBar} style={{ width: `${progress}%` }} />
        </div>
        <span className={styles.stepCount}>{current >= 0 ? current + 1 : 0} / {STEPS.length}</span>
      </div>

      {/* Info panel */}
      <div className={styles.infoPanel}>{infoText}</div>
    </div>
  );
}

/* ---- Sub-component ---- */
type FlowNodeProps = {
  step: typeof STEPS[0];
  state: StepState;
  onClick: () => void;
  decision?: boolean;
};

const ICON_COLORS: Record<string, { bg: string; color: string }> = {
  n1:  { bg: 'rgba(34,197,94,0.12)',  color: '#16a34a' },
  n2:  { bg: 'rgba(59,130,246,0.1)',  color: '#2563eb' },
  n3:  { bg: 'rgba(168,85,247,0.1)', color: '#7c3aed' },
  n4:  { bg: 'rgba(245,158,11,0.1)', color: '#b45309' },
  n5:  { bg: 'rgba(20,184,166,0.1)', color: '#0f766e' },
  n6:  { bg: 'rgba(34,197,94,0.12)', color: '#16a34a' },
  n7:  { bg: 'rgba(59,130,246,0.1)', color: '#2563eb' },
  n8:  { bg: 'rgba(168,85,247,0.1)', color: '#7c3aed' },
  n9:  { bg: 'rgba(245,158,11,0.1)', color: '#b45309' },
  n10: { bg: 'rgba(20,184,166,0.1)', color: '#0f766e' },
  n11: { bg: 'rgba(34,197,94,0.12)', color: '#16a34a' },
  n12: { bg: 'rgba(34,197,94,0.12)', color: '#16a34a' },
  n13: { bg: 'rgba(220,38,38,0.1)',  color: '#dc2626' },
  n14: { bg: 'rgba(59,130,246,0.1)', color: '#2563eb' },
};

function FlowNode({ step, state, onClick, decision }: FlowNodeProps) {
  const iconStyle = ICON_COLORS[step.id] || { bg: 'rgba(100,100,100,0.1)', color: '#555' };
  return (
    <div
      className={[
        styles.node,
        state === 'active' ? styles.nodeActive : '',
        state === 'done'   ? styles.nodeDone   : '',
        decision           ? styles.nodeDecision : '',
      ].join(' ')}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onClick()}
    >
      <div className={styles.nodeIcon} style={{ background: iconStyle.bg, color: iconStyle.color }}>
        {step.icon}
      </div>
      <div>
        <div className={styles.nodeTitle}>{step.title}</div>
        <div className={styles.nodeSub}>{step.sub}</div>
      </div>
    </div>
  );
}
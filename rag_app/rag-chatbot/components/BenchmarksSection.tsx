'use client';

import { useState } from 'react';
import styles from './BenchmarksSection.module.css';

type BenchItem = { label: string; score: number; top?: boolean };
type BenchKey = 'overall' | 'speed' | 'reading' | 'table';

const TABS: { key: BenchKey; label: string }[] = [
  { key: 'overall', label: 'Overall Score' },
  { key: 'speed',   label: 'Speed (s/page)' },
  { key: 'reading', label: 'Reading Order' },
  { key: 'table',   label: 'Table Score' },
];

const BENCH_DATA: Record<BenchKey, BenchItem[]> = {
  overall: [
    { label: 'ragchatbot [hybrid]', score: 0.907, top: true },
    { label: 'nutrient',            score: 0.885 },
    { label: 'docling',             score: 0.882 },
    { label: 'marker',              score: 0.861 },
    { label: 'unstructured [hi_res]', score: 0.841 },
    { label: 'ragchatbot',          score: 0.831, top: true },
    { label: 'mineru',              score: 0.831 },
    { label: 'pymupdf4llm',         score: 0.732 },
  ],
  speed: [
    { label: 'nutrient',            score: 0.008 },
    { label: 'ragchatbot',          score: 0.015, top: true },
    { label: 'unstructured',        score: 0.077 },
    { label: 'pymupdf4llm',         score: 0.091 },
    { label: 'ragchatbot [hybrid]', score: 0.463, top: true },
    { label: 'docling',             score: 0.762 },
  ],
  reading: [
    { label: 'ragchatbot [hybrid]', score: 0.934, top: true },
    { label: 'nutrient',            score: 0.925 },
    { label: 'unstructured [hi_res]', score: 0.904 },
    { label: 'ragchatbot',          score: 0.902, top: true },
    { label: 'docling',             score: 0.898 },
    { label: 'marker',              score: 0.890 },
  ],
  table: [
    { label: 'ragchatbot [hybrid]', score: 0.928, top: true },
    { label: 'docling',             score: 0.887 },
    { label: 'mineru',              score: 0.873 },
    { label: 'marker',              score: 0.808 },
    { label: 'ragchatbot',          score: 0.489, top: true },
    { label: 'pymupdf4llm',         score: 0.401 },
  ],
};

export default function BenchmarksSection() {
  const [active, setActive] = useState<BenchKey>('overall');
  const data = BENCH_DATA[active];
  const isSpeed = active === 'speed';
  const maxVal = isSpeed ? Math.max(...data.map((d) => d.score)) : 1;

  const getWidth = (score: number) =>
    isSpeed
      ? ((maxVal - score) / maxVal) * 85 + 10
      : score * 100;

  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <p className={styles.label}>Benchmarks</p>
        <h2 className={styles.title}>Why This RAG Stack?</h2>

        <div className={styles.tabs}>
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`${styles.tab} ${active === t.key ? styles.tabActive : ''}`}
              onClick={() => setActive(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className={styles.list}>
          {data.map((item) => (
            <div key={item.label} className={styles.row}>
              <span className={`${styles.rowLabel} ${item.top ? styles.rowLabelTop : ''}`}>
                {item.label}
              </span>
              <div className={styles.barWrap}>
                <div
                  className={`${styles.bar} ${item.top ? styles.barTop : ''}`}
                  style={{ width: `${getWidth(item.score)}%` }}
                />
              </div>
              <span className={styles.score}>{item.score}</span>
            </div>
          ))}
        </div>

        <div className={styles.footer}>
          <a href="/docs/benchmark" className={styles.link}>
            See transparent benchmark methodology →
          </a>
        </div>
      </div>
    </section>
  );
}
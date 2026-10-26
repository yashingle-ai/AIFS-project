import Link from 'next/link';
import styles from './SolutionSection.module.css';

const SOLUTIONS = [
  { title: 'Smart Reading Order', desc: 'Correctly reads multi-column layouts. Text flows in the order humans read it.', href: '/docs/reading-order' },
  { title: 'Hybrid OCR & AI', desc: 'Optional LLM enhancement for OCR and complex tables. 93% table accuracy in benchmarks.', href: '/docs/hybrid-mode' },
  { title: 'Bounding Boxes', desc: 'Every element includes [x1, y1, x2, y2] coordinates for precise citations and source linking.', href: '/docs/json-schema' },
  { title: 'Table Extraction', desc: 'Detects borders and clusters text into rows/columns. Handles merged cells gracefully.', href: '/docs/tables' },
  { title: 'Auto-Tagging Pipeline', desc: 'Untagged PDF in → screen-reader-ready Tagged PDF out. Based on PDF Association specs.', href: '/accessibility' },
  { title: 'AI Safety Built-in', desc: 'Filters hidden text, off-page content, and prompt injection attempts automatically.', href: '/docs/ai-safety' },
];

export default function SolutionSection() {
  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <p className={`${styles.label} ${styles.labelAccent}`}>The Solution</p>
        <h2 className={styles.title}>Built for RAG, Not Just Reading</h2>
        <p className={styles.sub}>
          Your RAG Chatbot delivers what LLM pipelines actually need — structure, accuracy, and
          citations.
        </p>

        <div className={styles.grid}>
          {SOLUTIONS.map((s) => (
            <div key={s.title} className={styles.card}>
              <h3 className={styles.cardTitle}>{s.title}</h3>
              <p className={styles.cardDesc}>{s.desc}</p>
              <Link href={s.href} className={styles.cardLink}>
                Learn more →
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
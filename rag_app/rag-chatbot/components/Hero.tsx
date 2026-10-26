import Link from 'next/link';
import styles from './Hero.module.css';
import RagFlowChart from './RagFlowChart';

export default function Hero() {
  return (
    <>
      <section className={styles.hero}>
        <div className={styles.topTags}>
          <span className={`${styles.tag} ${styles.tagAccent}`}>🏆 Top Benchmark Scores</span>
          <span className={styles.tag}>Apache-2.0</span>
        </div>

        <h1 className={styles.heading}>
          RAG Chatbot <em>Built</em>
          <br />
          for Your Data
        </h1>

        <p className={styles.sub}>
          Upload your documents, ask questions, get grounded answers.
          Retrieval-Augmented Generation with citations, source highlighting,
          and full traceability.
        </p>

        <div className={styles.featureTags}>
          <span>Semantic Search</span>
          <span className={styles.dot}>·</span>
          <span>OCR (80+ Languages)</span>
          <span className={styles.dot}>·</span>
          <span>Tables · Formulas · Charts</span>
        </div>

        <div className={styles.ctaRow}>
          <Link href="/demo" className={`${styles.btn} ${styles.btnPrimary}`}>
            Get Started →
          </Link>
          <Link href="/demo" className={`${styles.btn} ${styles.btnGhost}`}>
            Live Demo
          </Link>
          <Link href="/pdf" className={`${styles.btn} ${styles.btnOutline}`}>
            PDF Parsing
          </Link>
          <a
            href="https://github.com/your-org/rag-chatbot"
            target="_blank"
            rel="noopener noreferrer"
            className={`${styles.btn} ${styles.btnOutline}`}
          >
            GitHub ↗
          </a>
        </div>
      </section>

      {/* Hero media */}
      <div className={styles.mediaWrap}>
        <div className={styles.media}>
          <RagFlowChart />
        </div>
      </div>
    </>
  );
}
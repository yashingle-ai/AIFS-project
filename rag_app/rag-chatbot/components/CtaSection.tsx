import Link from 'next/link';
import styles from './CtaSection.module.css';

export default function CtaSection() {
  return (
    <section className={styles.section}>
      <p className={styles.label}>Get Started in Seconds</p>
      <h2 className={styles.title}>Ready to Build Your RAG Chatbot?</h2>
      <p className={styles.sub}>One command to get started. No API keys for parsing, no cloud, no hassle.</p>

      <div className={styles.terminal}>
        <span className={styles.prompt}>$</span>
        <span>pip install -U rag-chatbot-sdk</span>
      </div>

      <div className={styles.ctaRow}>
        <Link href="/docs" className={`${styles.btn} ${styles.btnPrimary}`}>
          Read the Docs
        </Link>
        <a
          href="https://github.com/your-org/rag-chatbot"
          target="_blank"
          rel="noopener noreferrer"
          className={`${styles.btn} ${styles.btnGhost}`}
        >
          View on GitHub ↗
        </a>
      </div>
    </section>
  );
}
import styles from './ProblemSection.module.css';

const PROBLEMS = [
  {
    icon: '🌀',
    title: 'Scrambled Reading Order',
    desc: 'Multi-column layouts read left-to-right across the page, mixing content. Your LLM receives jumbled text.',
  },
  {
    icon: '📉',
    title: 'Lost Table Structure',
    desc: 'Tables become walls of unformatted text. Row and column relationships disappear entirely.',
  },
  {
    icon: '📍',
    title: 'No Source Coordinates',
    desc: 'No way to cite where information came from or highlight the original document location.',
  },
  {
    icon: '♿',
    title: 'Accessibility Non-Compliance',
    desc: "EAA, ADA, Section 508 enforced worldwide. Manual document remediation doesn't scale.",
  },
];

export default function ProblemSection() {
  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <p className={styles.label}>The Problem</p>
        <h2 className={styles.title}>Documents Weren't Built for AI</h2>
        <p className={styles.sub}>
          Lost structure, broken tables, missing context — the parser you choose determines 90% of
          your chatbot's answer quality.
        </p>

        <blockquote className={styles.quote}>
          "If the data isn't parsed properly, your RAG system will never retrieve accurate answers.
          Garbage in = garbage out."
        </blockquote>

        <div className={styles.grid}>
          {PROBLEMS.map((p) => (
            <div key={p.title} className={styles.card}>
              <div className={styles.cardIcon}>{p.icon}</div>
              <h3 className={styles.cardTitle}>{p.title}</h3>
              <p className={styles.cardDesc}>{p.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
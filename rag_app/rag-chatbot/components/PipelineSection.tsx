import styles from './PipelineSection.module.css';

const STEPS = [
  { num: '1', tier: 'Free', title: 'Ingest & Audit', desc: 'Upload PDFs, check existing structure, detect untagged or malformed documents.', badge: 'Shipped', active: true },
  { num: '2', tier: 'Free (Apache 2.0)', title: 'Parse & Chunk', desc: 'Generate structured JSON with bounding boxes. Chunk intelligently for vector stores.', badge: 'Available', active: true },
  { num: '3', tier: 'Enterprise', title: 'Embed & Index', desc: 'Vector embeddings with metadata — page number, bounding box, source file.', badge: 'Available', active: false },
  { num: '4', tier: 'Enterprise', title: 'Chat with Citations', desc: 'Ask questions, get grounded answers with highlighted source passages.', badge: 'Available', active: false },
];

export default function PipelineSection() {
  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <p className={styles.label}>Pipeline</p>
        <h2 className={styles.title}>End-to-End Document Pipeline</h2>
        <p className={styles.sub}>From raw documents to AI-ready answers in four steps.</p>

        <div className={styles.steps}>
          {STEPS.map((s) => (
            <div key={s.num} className={styles.step}>
              <div className={`${styles.stepNum} ${s.active ? styles.stepNumActive : ''}`}>
                {s.num}
              </div>
              <div className={styles.stepBody}>
                <p className={`${styles.tier} ${s.active ? styles.tierFree : ''}`}>{s.tier}</p>
                <h3 className={styles.stepTitle}>{s.title}</h3>
                <p className={styles.stepDesc}>{s.desc}</p>
              </div>
              <span className={styles.badge}>{s.badge}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
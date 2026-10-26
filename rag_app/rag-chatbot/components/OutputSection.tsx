import styles from './OutputSection.module.css';

const FIELDS = [
  { key: 'type', desc: 'Element type: heading, paragraph, table, list, image' },
  { key: 'id', desc: 'Unique identifier for cross-referencing' },
  { key: 'page_number', desc: '1-indexed page reference' },
  { key: 'bounding_box', desc: '[left, bottom, right, top] in PDF points' },
  { key: 'heading_level', desc: 'Heading depth (1+)' },
  { key: 'font', desc: 'Typography info' },
  { key: 'content', desc: 'Extracted text' },
];

export default function OutputSection() {
  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <p className={styles.label}>Output Format</p>
        <h2 className={styles.title}>Structured Output with Bounding Boxes</h2>

        <div className={styles.grid}>
          {/* JSON block */}
          <div className={styles.codeBlock}>
            <div className={styles.codeHeader}>
              <span className={styles.dotR} /><span className={styles.dotY} /><span className={styles.dotG} />
              <span className={styles.codeTitle}>JSON Output Example</span>
            </div>
            <pre className={styles.pre}>{`{
  `}<span className={styles.key}>"type"</span>{`: `}<span className={styles.str}>"heading"</span>{`,
  `}<span className={styles.key}>"id"</span>{`: `}<span className={styles.num}>42</span>{`,
  `}<span className={styles.key}>"level"</span>{`: `}<span className={styles.str}>"Title"</span>{`,
  `}<span className={styles.key}>"page_number"</span>{`: `}<span className={styles.num}>1</span>{`,
  `}<span className={styles.key}>"bounding_box"</span>{`: [`}<span className={styles.num}>72.0</span>{`, `}<span className={styles.num}>700.0</span>{`,
    `}<span className={styles.num}>540.0</span>{`, `}<span className={styles.num}>730.0</span>{`],
  `}<span className={styles.key}>"heading_level"</span>{`: `}<span className={styles.num}>1</span>{`,
  `}<span className={styles.key}>"font"</span>{`: `}<span className={styles.str}>"Helvetica-Bold"</span>{`,
  `}<span className={styles.key}>"font_size"</span>{`: `}<span className={styles.num}>24.0</span>{`,
  `}<span className={styles.key}>"content"</span>{`: `}<span className={styles.str}>"Introduction"</span>{`
}`}</pre>
          </div>

          {/* Table block */}
          <div className={styles.codeBlock}>
            <div className={styles.codeHeader}>
              <span className={styles.dotR} /><span className={styles.dotY} /><span className={styles.dotG} />
              <span className={styles.codeTitle}>Field Reference</span>
            </div>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Field</th>
                  <th className={styles.th}>Description</th>
                </tr>
              </thead>
              <tbody>
                {FIELDS.map((f) => (
                  <tr key={f.key}>
                    <td className={`${styles.td} ${styles.tdKey}`}>{f.key}</td>
                    <td className={styles.td}>{f.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className={styles.infoBox}>
          <p className={styles.infoTitle}>Why Bounding Boxes Matter for RAG</p>
          <p className={styles.infoText}>
            When your LLM answers a question, bounding boxes let you highlight the exact source
            location in the document, build citation links with page and position references, and
            verify extraction accuracy by visual comparison.
          </p>
        </div>
      </div>
    </section>
  );
}
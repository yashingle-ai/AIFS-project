'use client';

import { useState } from 'react';
import Link from 'next/link';
import styles from './QuickStartSection.module.css';

type Lang = 'python' | 'node' | 'docker';

const CODE: Record<Lang, { install: string; main: string; filename: string }> = {
  python: {
    install: '$ pip install -U rag-chatbot-sdk',
    filename: 'main.py',
    main: `import rag_chatbot

rag_chatbot.ingest(
    input_path=["document.pdf"],
    output_dir="vector_store/",
    format="json,markdown"
)`,
  },
  node: {
    install: '$ npm install rag-chatbot-sdk',
    filename: 'index.js',
    main: `const { ingest } = require('rag-chatbot-sdk');

await ingest({
  inputPath: ['document.pdf'],
  outputDir: 'vector_store/',
  format: 'json,markdown',
});`,
  },
  docker: {
    install: '$ docker pull ragchatbot/sdk:latest',
    filename: 'docker-compose.yml',
    main: `services:
  rag:
    image: ragchatbot/sdk:latest
    volumes:
      - ./docs:/input
      - ./out:/output
    environment:
      FORMAT: "json,markdown"`,
  },
};

export default function QuickStartSection() {
  const [lang, setLang] = useState<Lang>('python');
  const c = CODE[lang];

  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <p className={styles.label}>Quick Start</p>
        <h2 className={styles.title}>Get Started in 60 Seconds</h2>

        <div className={styles.getStartedButtonContainer}>
          <Link href="/demo" className={styles.getStartedButton}>
            Try the Demo
          </Link>
        </div>

        <div className={styles.tabs}>
          {(['python', 'node', 'docker'] as Lang[]).map((l) => (
            <button
              key={l}
              className={`${styles.tab} ${lang === l ? styles.tabActive : ''}`}
              onClick={() => setLang(l)}
            >
              {l === 'python' ? 'Python' : l === 'node' ? 'Node.js' : 'Docker'}
            </button>
          ))}
        </div>

        <div className={styles.codeBlock}>
          <div className={styles.codeHeader}>
            <span className={styles.dotR} /><span className={styles.dotY} /><span className={styles.dotG} />
            <span className={styles.filename}>terminal</span>
          </div>
          <pre className={styles.pre}>
            <span className={styles.prompt}>$</span>{' '}
            {c.install.replace('$ ', '')}
          </pre>
        </div>

        <div className={`${styles.codeBlock} ${styles.mt}`}>
          <div className={styles.codeHeader}>
            <span className={styles.dotR} /><span className={styles.dotY} /><span className={styles.dotG} />
            <span className={styles.filename}>{c.filename}</span>
          </div>
          <pre className={styles.pre}>{c.main}</pre>
        </div>

        <div className={styles.ragBox}>
          <p className={styles.ragTitle}>Building a RAG pipeline?</p>
          <p className={styles.ragSub}>Use our official LangChain integration:</p>
          <div className={styles.codeBlock}>
            <pre className={styles.pre}>
              <span className={styles.prompt}>$</span> pip install -U langchain-rag-chatbot
            </pre>
          </div>
          <a href="/docs/rag-integration" className={styles.ragLink}>
            View RAG Integration Guide →
          </a>
        </div>
      </div>
    </section>
  );
}
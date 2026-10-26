import Link from 'next/link';
import styles from './Footer.module.css';

const LINKS = [
  { label: 'Documentation', href: '/docs' },
  { label: 'Demo', href: '/demo' },
  { label: 'Contact', href: '/contact' },
  { label: 'Privacy Policy', href: '/privacy' },
];

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <span className={styles.copy}>© {new Date().getFullYear()} RAGChatbot App</span>
      <nav className={styles.links}>
        {LINKS.map((l) => (
          <Link key={l.href} href={l.href} className={styles.link}>
            {l.label}
          </Link>
        ))}
      </nav>
    </footer>
  );
}
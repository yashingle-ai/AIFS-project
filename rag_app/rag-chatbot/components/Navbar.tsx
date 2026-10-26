'use client';

import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import styles from './Navbar.module.css';

const NAV_LINKS = [
  { label: 'Documentation', href: '/docs' },
  { label: 'Features', href: '/features' },
  { label: 'Demo', href: '/demo' },
  { label: 'Showcase', href: '/showcase' },
  { label: 'Contact', href: '/contact' },
];

export default function Navbar() {
  const { currentUser, ready, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleEscape);
    };
  }, []);

  const initials = useMemo(() => {
    if (!currentUser) {
      return 'U';
    }

    const firstInitial = currentUser.firstName.trim().charAt(0);
    const lastInitial = currentUser.lastName.trim().charAt(0);
    return `${firstInitial}${lastInitial}`.trim().toUpperCase() || currentUser.email.slice(0, 2).toUpperCase();
  }, [currentUser]);

  return (
    <nav className={styles.nav}>
      <Link href="/" className={styles.logo}>
        <div className={styles.logoIcon}>R</div>
        RAGChatbot
      </Link>

      <div className={styles.links}>
        {NAV_LINKS.map((link) => (
          <Link key={link.href} href={link.href} className={styles.navLink}>
            {link.label}
          </Link>
        ))}
      </div>

      <div className={styles.right}>
        <div className={styles.searchBox}>
          Search <kbd className={styles.kbd}>⌘</kbd><kbd className={styles.kbd}>K</kbd>
        </div>

        {!ready ? (
          <div className={styles.authPlaceholder} />
        ) : currentUser ? (
          <div ref={menuRef} className={styles.profileWrap}>
            <button
              type="button"
              className={styles.profileButton}
              onClick={() => setMenuOpen((value) => !value)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              aria-label="Open profile menu"
            >
              {initials}
            </button>

            {menuOpen ? (
              <div className={styles.profileMenu} role="menu" aria-label="Profile menu">
                <Link href="/profile" className={styles.menuItem} role="menuitem" onClick={() => setMenuOpen(false)}>
                  Profile
                </Link>
                <button type="button" className={styles.menuItem} onClick={() => { logout(); setMenuOpen(false); }}>
                  Log out
                </button>
              </div>
            ) : null}
          </div>
        ) : (
          <div className={styles.authActions}>
            <Link href="/login" className={styles.authLinkSecondary}>
              Login
            </Link>
            <Link href="/signup" className={styles.authLinkPrimary}>
              Sign up
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
}
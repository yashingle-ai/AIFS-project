'use client';

import Link from 'next/link';
import type { ReactNode } from 'react';
import styles from './AuthShell.module.css';

type AuthShellProps = {
    eyebrow: string;
    title: string;
    description: string;
    footerText?: string;
    footerLinkText?: string;
    footerLinkHref?: string;
    children: ReactNode;
};

export default function AuthShell({
    eyebrow,
    title,
    description,
    footerText,
    footerLinkText,
    footerLinkHref,
    children,
}: AuthShellProps) {
    return (
        <main className={styles.page}>
            <div className={styles.glowA} />
            <div className={styles.glowB} />

            <section className={styles.shell}>
                <div className={styles.copy}>
                    <p className={styles.eyebrow}>{eyebrow}</p>
                    <h1 className={styles.title}>{title}</h1>
                    <p className={styles.description}>{description}</p>
                </div>

                <div className={styles.card}>{children}</div>
            </section>

            {footerText && footerLinkText && footerLinkHref ? (
                <p className={styles.footer}>
                    {footerText}{' '}
                    <Link href={footerLinkHref} className={styles.footerLink}>
                        {footerLinkText}
                    </Link>
                </p>
            ) : null}
        </main>
    );
}
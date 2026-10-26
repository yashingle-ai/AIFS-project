'use client';

import Link from 'next/link';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '@/components/Navbar';
import AuthShell from '@/components/AuthShell';
import { useAuth } from '@/context/AuthContext';
import styles from '@/components/AuthShell.module.css';

function getInitials(firstName: string, lastName: string, email: string) {
    const firstInitial = firstName.trim().charAt(0);
    const lastInitial = lastName.trim().charAt(0);
    return `${firstInitial}${lastInitial}`.trim().toUpperCase() || email.slice(0, 2).toUpperCase();
}

export default function ProfilePage() {
    const router = useRouter();
    const { currentUser, ready, logout } = useAuth();

    useEffect(() => {
        if (ready && !currentUser) {
            router.replace('/login');
        }
    }, [currentUser, ready, router]);

    if (!ready || !currentUser) {
        return (
            <>
                <Navbar />
                <AuthShell
                    eyebrow="Loading profile"
                    title="Preparing your account"
                    description="Your session is being restored. If you are not signed in, we will send you back to the login page."
                    footerText=""
                    footerLinkText=""
                    footerLinkHref="/login"
                >
                    <p className={styles.panelText}>Loading...</p>
                </AuthShell>
            </>
        );
    }

    const initials = getInitials(currentUser.firstName, currentUser.lastName, currentUser.email);

    return (
        <>
            <Navbar />
            <AuthShell
                eyebrow="Account profile"
                title="Your profile"
                description="This profile card reflects the signed-in user and uses the same accent, spacing, and typography as the rest of the frontend."
                footerText="Back to the"
                footerLinkText="homepage"
                footerLinkHref="/"
            >
                <div className={styles.profileBox}>
                    <div className={styles.profileAvatar}>{initials}</div>

                    <div className={styles.profileMeta}>
                        <div className={styles.metaRow}>
                            <span className={styles.metaLabel}>First name</span>
                            <span className={styles.metaValue}>{currentUser.firstName}</span>
                        </div>

                        <div className={styles.metaRow}>
                            <span className={styles.metaLabel}>Last name</span>
                            <span className={styles.metaValue}>{currentUser.lastName}</span>
                        </div>

                        <div className={styles.metaRow}>
                            <span className={styles.metaLabel}>Email</span>
                            <span className={styles.metaValue}>{currentUser.email}</span>
                        </div>
                    </div>

                    <button className={styles.secondaryAction} type="button" onClick={logout}>
                        Log out
                    </button>

                    <Link href="/" className={styles.secondaryAction}>
                        Return to home
                    </Link>
                </div>
            </AuthShell>
        </>
    );
}
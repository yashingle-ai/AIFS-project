'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Navbar from '@/components/Navbar';
import AuthShell from '@/components/AuthShell';
import { useAuth } from '@/context/AuthContext';
import styles from '@/components/AuthShell.module.css';
import { normalizeInternalPath } from '@/lib/navigation';

function SignupPageContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const { currentUser, ready, signup } = useAuth();
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const returnTo = normalizeInternalPath(searchParams.get('returnTo'), '/profile');

    useEffect(() => {
        if (ready && currentUser) {
            router.replace(returnTo);
        }
    }, [currentUser, ready, returnTo, router]);

    const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError('');

        try {
            signup({ firstName, lastName, email, password });
            router.replace(returnTo);
        } catch (submissionError) {
            setError(submissionError instanceof Error ? submissionError.message : 'Unable to create your account.');
        }
    };

    return (
        <>
            <Navbar />
            <AuthShell
                eyebrow="Join the workspace"
                title="Create your account"
                description="Sign up once and your initials, profile, and session stay inside the same dark green visual language as the rest of the site."
                footerText="Already have an account?"
                footerLinkText="Log in instead"
                footerLinkHref={`/login?returnTo=${encodeURIComponent(returnTo)}`}
            >
                <div>
                    <p className={styles.panelTitle}>Sign up</p>
                    <p className={styles.panelText}>Enter your first name, last name, email, and password to get started.</p>

                    <form className={styles.form} onSubmit={handleSubmit}>
                        <div className={styles.row}>
                            <label className={styles.label}>
                                First name
                                <input
                                    className={styles.input}
                                    type="text"
                                    value={firstName}
                                    onChange={(event) => setFirstName(event.target.value)}
                                    placeholder="First name"
                                    required
                                />
                            </label>

                            <label className={styles.label}>
                                Last name
                                <input
                                    className={styles.input}
                                    type="text"
                                    value={lastName}
                                    onChange={(event) => setLastName(event.target.value)}
                                    placeholder="Last name"
                                    required
                                />
                            </label>
                        </div>

                        <label className={styles.label}>
                            Email
                            <input
                                className={styles.input}
                                type="email"
                                value={email}
                                onChange={(event) => setEmail(event.target.value)}
                                placeholder="name@example.com"
                                required
                            />
                        </label>

                        <label className={styles.label}>
                            Password
                            <input
                                className={styles.input}
                                type="password"
                                value={password}
                                onChange={(event) => setPassword(event.target.value)}
                                placeholder="Create a password"
                                required
                            />
                        </label>

                        {error ? <p className={styles.error}>{error}</p> : null}

                        <button className={styles.submit} type="submit">
                            Create account
                        </button>
                    </form>
                </div>
            </AuthShell>
        </>
    );
}

export default function SignupPage() {
    return (
        <Suspense fallback={null}>
            <SignupPageContent />
        </Suspense>
    );
}
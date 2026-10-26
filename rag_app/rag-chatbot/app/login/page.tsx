'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Navbar from '@/components/Navbar';
import AuthShell from '@/components/AuthShell';
import { useAuth } from '@/context/AuthContext';
import styles from '@/components/AuthShell.module.css';
import { normalizeInternalPath } from '@/lib/navigation';

function LoginPageContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const { currentUser, ready, login } = useAuth();
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
            login({ email, password });
            router.replace(returnTo);
        } catch (submissionError) {
            setError(submissionError instanceof Error ? submissionError.message : 'Unable to log in.');
        }
    };

    return (
        <>
            <Navbar />
            <AuthShell
                eyebrow="Secure access"
                title="Welcome back"
                description="Log in to resume your saved conversations, keep your profile handy, and continue inside the same visual system as the rest of the site."
                footerText="Need an account?"
                footerLinkText="Create one here"
                footerLinkHref={`/signup?returnTo=${encodeURIComponent(returnTo)}`}
            >
                <div>
                    <p className={styles.panelTitle}>Login</p>
                    <p className={styles.panelText}>Use the email and password you registered with.</p>

                    <form className={styles.form} onSubmit={handleSubmit}>
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
                                placeholder="Enter your password"
                                required
                            />
                        </label>

                        {error ? <p className={styles.error}>{error}</p> : null}

                        <button className={styles.submit} type="submit">
                            Log in
                        </button>
                    </form>
                </div>
            </AuthShell>
        </>
    );
}

export default function LoginPage() {
    return (
        <Suspense fallback={null}>
            <LoginPageContent />
        </Suspense>
    );
}
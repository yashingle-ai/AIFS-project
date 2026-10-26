'use client';

import {
    createContext,
    useContext,
    useEffect,
    useState,
    type ReactNode,
} from 'react';

export type AuthUser = {
    firstName: string;
    lastName: string;
    email: string;
    password: string;
};

type AuthContextValue = {
    currentUser: AuthUser | null;
    ready: boolean;
    login: (credentials: Pick<AuthUser, 'email' | 'password'>) => void;
    signup: (user: AuthUser) => void;
    logout: () => void;
};

const USERS_STORAGE_KEY = 'ragchatbot-users';
const SESSION_STORAGE_KEY = 'ragchatbot-session';

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function readUsers(): AuthUser[] {
    if (typeof window === 'undefined') {
        return [];
    }

    try {
        const rawValue = window.localStorage.getItem(USERS_STORAGE_KEY);
        return rawValue ? (JSON.parse(rawValue) as AuthUser[]) : [];
    } catch {
        return [];
    }
}

export function AuthProvider({ children }: { children: ReactNode }) {
    const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
    const [ready, setReady] = useState(false);

    useEffect(() => {
        const storedUsers = readUsers();

        try {
            const sessionEmail = window.localStorage.getItem(SESSION_STORAGE_KEY);
            if (sessionEmail) {
                const storedUser = storedUsers.find((user) => user.email === sessionEmail);
                setCurrentUser(storedUser ?? null);
            }
        } finally {
            setReady(true);
        }
    }, []);

    const persistUsers = (users: AuthUser[]) => {
        window.localStorage.setItem(USERS_STORAGE_KEY, JSON.stringify(users));
    };

    const login = (credentials: Pick<AuthUser, 'email' | 'password'>) => {
        const storedUsers = readUsers();
        const matchedUser = storedUsers.find(
            (user) => user.email.toLowerCase() === credentials.email.toLowerCase() && user.password === credentials.password,
        );

        if (!matchedUser) {
            throw new Error('Invalid email or password.');
        }

        window.localStorage.setItem(SESSION_STORAGE_KEY, matchedUser.email);
        setCurrentUser(matchedUser);
    };

    const signup = (user: AuthUser) => {
        const storedUsers = readUsers();
        const existingUser = storedUsers.find(
            (item) => item.email.toLowerCase() === user.email.toLowerCase(),
        );

        if (existingUser) {
            throw new Error('An account with this email already exists.');
        }

        const nextUsers = [...storedUsers, user];
        persistUsers(nextUsers);
        window.localStorage.setItem(SESSION_STORAGE_KEY, user.email);
        setCurrentUser(user);
    };

    const logout = () => {
        window.localStorage.removeItem(SESSION_STORAGE_KEY);
        setCurrentUser(null);
    };

    return (
        <AuthContext.Provider value={{ currentUser, ready, login, signup, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);

    if (!context) {
        throw new Error('useAuth must be used inside AuthProvider.');
    }

    return context;
}
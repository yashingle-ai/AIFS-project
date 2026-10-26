const DEFAULT_API_BASE_URL = 'http://localhost:8000';

export const RAG_API_BASE_URL = process.env.NEXT_PUBLIC_RAG_API_BASE_URL ?? DEFAULT_API_BASE_URL;

export function apiUrl(path: string) {
    return new URL(path, RAG_API_BASE_URL).toString();
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(apiUrl(path), {
        ...init,
        headers: {
            Accept: 'application/json',
            ...(init?.headers ?? {}),
        },
    });

    if (!response.ok) {
        let message = `Request failed (${response.status})`;

        try {
            const responseText = await response.text();
            if (responseText) {
                message = responseText;
            }
        } catch {
            // Ignore secondary parsing failures and use the fallback message.
        }

        throw new Error(message);
    }

    return response.json() as Promise<T>;
}
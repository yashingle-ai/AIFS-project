export function normalizeInternalPath(value: string | null, fallback: string) {
    if (!value) {
        return fallback;
    }

    if (!value.startsWith('/') || value.startsWith('//') || value.includes('://')) {
        return fallback;
    }

    return value;
}
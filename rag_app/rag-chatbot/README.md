# RAG Chatbot Frontend

This is the Next.js frontend for the document-grounded RAG chatbot. It connects to the FastAPI backend in the parent workspace and provides the landing page, auth screens, and the full chatbot demo with document upload, streaming answers, citations, and feedback.

## Features

- Landing page with a `Get Started` entry point into the chatbot demo.
- Document upload for PDF files, with indexing status and document management.
- Streaming chat responses with markdown rendering.
- Clickable citations and a retrieved-chunk side panel.
- Theme-matched UI that follows the dark green visual style of the website.

## Prerequisites

- Node.js 18+.
- The backend server running from the parent project.

The frontend expects the API at `http://localhost:8000` by default.

## Setup

From the `rag-chatbot` folder:

```bash
npm install
```

If your backend runs on a different host or port, create a `.env.local` file:

```bash
NEXT_PUBLIC_RAG_API_BASE_URL=http://localhost:8000
```

## Run the App

Start the frontend:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Backend

Before using the chatbot, start the FastAPI backend from the project root:

```bash
uvicorn server:app --reload --port 8000
```

The frontend uses these backend endpoints:

- `GET /api/info`
- `GET /api/documents`
- `POST /api/documents`
- `GET /api/documents/{doc_id}/suggestions`
- `POST /api/documents/{doc_id}/chat/stream`
- `POST /api/feedback`

## Main Routes

- `/` - marketing landing page.
- `/demo` - authenticated chatbot workspace.
- `/login` and `/signup` - local demo auth flow.
- `/profile` - basic user profile page.

## Project Structure

- `app/` - Next.js routes and page layouts.
- `components/` - shared landing page and UI components.
- `context/` - client-side auth state.
- `lib/` - API helpers and route utilities.

## Notes

- Chat answers are rendered as markdown.
- Clicking a citation opens the retrieved chunk details in the side panel.
- Clicking the page number opens the PDF directly at that page.

## Lint

```bash
npm run lint
```

If lint reports unrelated issues in other components, they are outside the chatbot integration changes.

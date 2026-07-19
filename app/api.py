"""FastAPI interface for testing the Noon RAG pipeline."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .rag.chain import answer_question


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceDocument(BaseModel):
    source: str
    part: int | None = None
    sub_part: int | None = None
    path: str | None = None
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    source_count: int
    sources: list[SourceDocument]


app = FastAPI(
    title="Noon RAG API",
    version="1.0.0",
    description="FastAPI test interface for the Noon customer support RAG system.",
)


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
        return HTMLResponse(
                """
                <!doctype html>
                <html lang="en">
                <head>
                    <meta charset="utf-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1" />
                    <title>Noon RAG Studio</title>
                    <style>
                        :root {
                            color-scheme: light;
                            --bg: #f6f1e8;
                            --bg-alt: #fffaf2;
                            --panel: rgba(255, 255, 255, 0.82);
                            --panel-border: rgba(35, 31, 32, 0.08);
                            --text: #1f1a17;
                            --muted: #6d645e;
                            --accent: #e0522d;
                            --accent-dark: #b43f20;
                            --success: #12734d;
                            --shadow: 0 24px 70px rgba(49, 32, 24, 0.12);
                        }

                        * { box-sizing: border-box; }
                        body {
                            margin: 0;
                            min-height: 100vh;
                            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                            color: var(--text);
                            background:
                                radial-gradient(circle at top left, rgba(224, 82, 45, 0.16), transparent 28%),
                                radial-gradient(circle at 85% 15%, rgba(18, 115, 77, 0.12), transparent 24%),
                                linear-gradient(180deg, #fffdf8 0%, var(--bg) 100%);
                        }

                        .shell {
                            max-width: 1120px;
                            margin: 0 auto;
                            padding: 32px 20px 48px;
                        }

                        .hero {
                            display: grid;
                            gap: 18px;
                            grid-template-columns: 1.3fr 0.7fr;
                            align-items: end;
                            margin-bottom: 24px;
                        }

                        .eyebrow {
                            display: inline-flex;
                            align-items: center;
                            gap: 8px;
                            padding: 8px 12px;
                            border-radius: 999px;
                            background: rgba(224, 82, 45, 0.08);
                            color: var(--accent-dark);
                            font-size: 12px;
                            font-weight: 700;
                            letter-spacing: 0.08em;
                            text-transform: uppercase;
                            width: fit-content;
                        }

                        h1 {
                            margin: 12px 0 8px;
                            font-size: clamp(2.3rem, 4vw, 4.4rem);
                            line-height: 0.98;
                            letter-spacing: -0.04em;
                        }

                        .lede {
                            max-width: 62ch;
                            margin: 0;
                            color: var(--muted);
                            font-size: 1.03rem;
                            line-height: 1.7;
                        }

                        .stats {
                            display: grid;
                            grid-template-columns: repeat(3, minmax(0, 1fr));
                            gap: 12px;
                        }

                        .stat {
                            padding: 16px;
                            border-radius: 18px;
                            background: rgba(255, 255, 255, 0.7);
                            border: 1px solid var(--panel-border);
                            box-shadow: var(--shadow);
                        }

                        .stat strong {
                            display: block;
                            font-size: 1.15rem;
                            margin-bottom: 6px;
                        }

                        .stat span {
                            color: var(--muted);
                            font-size: 0.94rem;
                            line-height: 1.45;
                        }

                        .card {
                            background: var(--panel);
                            backdrop-filter: blur(12px);
                            border: 1px solid var(--panel-border);
                            border-radius: 28px;
                            box-shadow: var(--shadow);
                            overflow: hidden;
                        }

                        .card-head {
                            padding: 20px 22px;
                            border-bottom: 1px solid rgba(31, 26, 23, 0.08);
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            gap: 12px;
                            flex-wrap: wrap;
                        }

                        .card-head h2 {
                            margin: 0;
                            font-size: 1.1rem;
                        }

                        .card-head p {
                            margin: 4px 0 0;
                            color: var(--muted);
                            font-size: 0.95rem;
                        }

                        .content {
                            display: grid;
                            grid-template-columns: 1fr 1.1fr;
                            gap: 0;
                        }

                        .pane {
                            padding: 22px;
                        }

                        .pane + .pane {
                            border-left: 1px solid rgba(31, 26, 23, 0.08);
                        }

                        label {
                            display: block;
                            font-size: 0.92rem;
                            font-weight: 700;
                            margin-bottom: 8px;
                        }

                        textarea, input, select, button {
                            font: inherit;
                        }

                        textarea, input, select {
                            width: 100%;
                            border-radius: 16px;
                            border: 1px solid rgba(31, 26, 23, 0.12);
                            background: rgba(255, 255, 255, 0.9);
                            padding: 14px 16px;
                            color: var(--text);
                            outline: none;
                            transition: border-color 0.15s ease, box-shadow 0.15s ease;
                        }

                        textarea:focus, input:focus, select:focus {
                            border-color: rgba(224, 82, 45, 0.45);
                            box-shadow: 0 0 0 4px rgba(224, 82, 45, 0.12);
                        }

                        textarea {
                            min-height: 160px;
                            resize: vertical;
                        }

                        .row {
                            display: grid;
                            grid-template-columns: 1fr 140px;
                            gap: 12px;
                            margin-top: 14px;
                        }

                        .actions {
                            display: flex;
                            gap: 10px;
                            align-items: center;
                            margin-top: 16px;
                        }

                        button {
                            border: 0;
                            border-radius: 999px;
                            padding: 13px 18px;
                            cursor: pointer;
                            font-weight: 700;
                        }

                        .primary {
                            background: linear-gradient(135deg, var(--accent), #f07b24);
                            color: white;
                            box-shadow: 0 16px 30px rgba(224, 82, 45, 0.22);
                        }

                        .secondary {
                            background: rgba(31, 26, 23, 0.06);
                            color: var(--text);
                        }

                        .hint {
                            color: var(--muted);
                            font-size: 0.9rem;
                            margin-top: 10px;
                            line-height: 1.5;
                        }

                        .output {
                            display: grid;
                            gap: 14px;
                        }

                        .answer-box, .source-box, .error-box {
                            border-radius: 20px;
                            border: 1px solid rgba(31, 26, 23, 0.08);
                            padding: 16px;
                            background: rgba(255, 255, 255, 0.88);
                        }

                        .answer-box h3, .source-box h3, .error-box h3 {
                            margin: 0 0 10px;
                            font-size: 0.95rem;
                            text-transform: uppercase;
                            letter-spacing: 0.06em;
                        }

                        .answer-text {
                            white-space: pre-wrap;
                            line-height: 1.7;
                        }

                        .sources {
                            display: grid;
                            gap: 10px;
                        }

                        .source-item {
                            border-radius: 16px;
                            background: rgba(18, 115, 77, 0.06);
                            padding: 12px 14px;
                        }

                        .source-item strong {
                            display: block;
                            margin-bottom: 4px;
                        }

                        .meta {
                            color: var(--muted);
                            font-size: 0.88rem;
                        }

                        .status {
                            color: var(--success);
                            font-weight: 700;
                            font-size: 0.92rem;
                        }

                        .error-box {
                            display: none;
                            border-color: rgba(176, 42, 42, 0.18);
                            background: rgba(255, 242, 242, 0.95);
                            color: #8a1f1f;
                        }

                        .loading {
                            opacity: 0.72;
                            pointer-events: none;
                        }

                        @media (max-width: 920px) {
                            .hero, .content, .row {
                                grid-template-columns: 1fr;
                            }

                            .pane + .pane {
                                border-left: 0;
                                border-top: 1px solid rgba(31, 26, 23, 0.08);
                            }

                            .stats {
                                grid-template-columns: 1fr;
                            }
                        }
                    </style>
                </head>
                <body>
                    <main class="shell">
                        <section class="hero">
                            <div>
                                <div class="eyebrow">Noon RAG Studio</div>
                                <h1>Test your customer support RAG with a clean, fast interface.</h1>
                                <p class="lede">
                                    Ask product, shipping, refund, or gift card questions in English or Arabic.
                                    The app routes to the cloud Chroma knowledge base and answers through Gemini.
                                </p>
                            </div>
                            <div class="stats">
                                <div class="stat">
                                    <strong>FastAPI</strong>
                                    <span>Lightweight web UI for testing retrieval and answer quality.</span>
                                </div>
                                <div class="stat">
                                    <strong>Chroma Cloud</strong>
                                    <span>Uses your hosted collection for document retrieval.</span>
                                </div>
                                <div class="stat">
                                    <strong>Gemini</strong>
                                    <span>Generates longer, natural answers without a fixed token cap.</span>
                                </div>
                            </div>
                        </section>

                        <section class="card" id="app-card">
                            <div class="card-head">
                                <div>
                                    <h2>Ask the assistant</h2>
                                    <p>Use the form on the left and inspect the answer plus source passages on the right.</p>
                                </div>
                                <div class="status" id="status">Ready</div>
                            </div>

                            <div class="content">
                                <div class="pane">
                                    <label for="question">Question</label>
                                    <textarea id="question" placeholder="Example: I want a gift card from noon for a wedding. How do I get one?"></textarea>

                                    <div class="row">
                                        <div>
                                            <label for="top_k">Top K</label>
                                            <input id="top_k" type="number" min="1" max="20" value="5" />
                                        </div>
                                        <div>
                                            <label for="language">Language</label>
                                            <select id="language">
                                                <option value="auto">Auto</option>
                                                <option value="en">English</option>
                                                <option value="ar">Arabic</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="actions">
                                        <button class="primary" id="ask-btn" type="button" onclick="window.NoonRag.ask()">Ask RAG</button>
                                        <button class="secondary" id="example-btn" type="button" onclick="window.NoonRag.loadExample()">Load example</button>
                                    </div>

                                    <p class="hint">
                                        The UI calls <code>/ask</code> directly, so you can compare the answer here with the raw API response in the docs.
                                    </p>
                                </div>

                                <div class="pane output">
                                    <div class="error-box" id="error-box">
                                        <h3>Error</h3>
                                        <div id="error-text"></div>
                                    </div>

                                    <div class="answer-box">
                                        <h3>Answer</h3>
                                        <div class="answer-text" id="answer">Your result will appear here.</div>
                                    </div>

                                    <div class="source-box">
                                        <h3>Sources</h3>
                                        <div class="sources" id="sources"></div>
                                    </div>
                                </div>
                            </div>
                        </section>
                    </main>

                    <script>
                        window.NoonRag = (() => {
                            const questionEl = document.getElementById('question');
                            const topKEl = document.getElementById('top_k');
                            const languageEl = document.getElementById('language');
                            const askBtn = document.getElementById('ask-btn');
                            const exampleBtn = document.getElementById('example-btn');
                            const answerEl = document.getElementById('answer');
                            const sourcesEl = document.getElementById('sources');
                            const errorBox = document.getElementById('error-box');
                            const errorText = document.getElementById('error-text');
                            const statusEl = document.getElementById('status');
                            const card = document.getElementById('app-card');

                            const examples = {
                                en: "My friend's wedding is near, I want to get her a gift card from Noon as a wedding gift. How do I do this?",
                                ar: 'كيف أشتري بطاقة هدية من نون كهدية زواج لصديقتي؟',
                                auto: 'How can I cancel my order?'
                            };

                            function setBusy(isBusy) {
                                card.classList.toggle('loading', isBusy);
                                statusEl.textContent = isBusy ? 'Thinking...' : 'Ready';
                                askBtn.disabled = isBusy;
                                exampleBtn.disabled = isBusy;
                                askBtn.textContent = isBusy ? 'Working...' : 'Ask RAG';
                            }

                            function showError(message) {
                                errorText.textContent = message;
                                errorBox.style.display = 'block';
                            }

                            function clearError() {
                                errorText.textContent = '';
                                errorBox.style.display = 'none';
                            }

                            function renderSources(sources) {
                                if (!sources || !sources.length) {
                                    sourcesEl.innerHTML = '<div class="meta">No sources returned.</div>';
                                    return;
                                }

                                sourcesEl.innerHTML = sources.map((source) => {
                                    const parts = [source.source];
                                    if (source.part !== null && source.part !== undefined) parts.push('part ' + source.part);
                                    if (source.sub_part !== null && source.sub_part !== undefined) parts.push('sub ' + source.sub_part);
                                    if (source.path) parts.push(source.path);
                                    return '<div class="source-item"><strong>' + parts.join(' | ') + '</strong><div class="meta">' + (source.preview || '') + '</div></div>';
                                }).join('');
                            }

                            async function ask() {
                                clearError();
                                const question = questionEl.value.trim();
                                const top_k = Number(topKEl.value || 5);

                                if (!question) {
                                    showError('Please enter a question first.');
                                    return;
                                }

                                setBusy(true);

                                try {
                                    const response = await fetch('/ask', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ question, top_k }),
                                    });

                                    const payload = await response.json();

                                    if (!response.ok) {
                                        throw new Error(payload.detail || 'Request failed');
                                    }

                                    answerEl.textContent = payload.answer || '';
                                    renderSources(payload.sources || []);
                                    statusEl.textContent = 'Returned ' + (payload.source_count || 0) + ' source(s)';
                                } catch (error) {
                                    showError(error.message || String(error));
                                    answerEl.textContent = 'No answer returned.';
                                    sourcesEl.innerHTML = '';
                                    statusEl.textContent = 'Error';
                                } finally {
                                    setBusy(false);
                                }
                            }

                            function loadExample() {
                                const lang = languageEl.value;
                                questionEl.value = examples[lang] || examples.auto;
                                questionEl.focus();
                            }

                            questionEl.addEventListener('keydown', (event) => {
                                if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                                    ask();
                                }
                            });

                            sourcesEl.innerHTML = '<div class="meta">Run a question to see supporting sources here.</div>';

                            return { ask, loadExample };
                        })();
                    </script>
                </body>
                </html>
                """
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _serialize_result(result: dict) -> AskResponse:
    documents = result.get("documents", []) or []
    sources: list[SourceDocument] = []

    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        content = getattr(document, "page_content", "") or ""
        sources.append(
            SourceDocument(
                source=str(metadata.get("source", "unknown source")),
                part=metadata.get("part"),
                sub_part=metadata.get("sub_part"),
                path=metadata.get("path"),
                preview=" ".join(content.split())[:280],
            )
        )

    return AskResponse(
        question=str(result.get("question", "")).strip(),
        answer=str(result.get("answer", "")).strip(),
        source_count=len(sources),
        sources=sources,
    )


async def _ask(question: str, top_k: int) -> AskResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    try:
        result = await run_in_threadpool(answer_question, question, top_k)
        return _serialize_result(result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive HTTP boundary
        raise HTTPException(status_code=500, detail=f"RAG request failed: {exc}") from exc


@app.get("/ask", response_model=AskResponse)
async def ask_get(
    question: str = Query(..., min_length=1, max_length=4000),
    top_k: int = Query(default=5, ge=1, le=20),
) -> AskResponse:
    return await _ask(question, top_k)


@app.post("/ask", response_model=AskResponse)
async def ask_post(payload: AskRequest) -> AskResponse:
    return await _ask(payload.question, payload.top_k)
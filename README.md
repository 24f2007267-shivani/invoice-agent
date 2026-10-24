# Invoice Action Agent (FastAPI / A2A HTTP+JSON)

## Important deployment settings
- `A2A_BASE_URL`: exact public base URL submitted to the grader, including `/a2a/`, e.g. `https://your-service.onrender.com/a2a/`.
- `A2A_BASE_PATH`: `/a2a` for the route prefix.
- `AGENT_BEARER_TOKEN`: secret bearer token shared with the grader configuration. Do not commit it.
- `DATABASE_PATH`: persistent SQLite file path.
- `AI_API_KEY`, `AI_API_URL`, `AI_MODEL`: OpenAI-compatible chat-completions endpoint.

## Local run
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export AGENT_BEARER_TOKEN='replace-with-long-random-secret'
export A2A_BASE_URL='http://127.0.0.1:8000/a2a/'
export DATABASE_PATH='./invoice-agent.sqlite3'
export AI_API_KEY='your-provider-key'
uvicorn main:app --reload
```

The service fails closed when no AI key is configured; no fabricated decisions are returned.

## Render
1. Create a new GitHub repository and push these files.
2. Create a Render Blueprint from the repository (or create a Web Service with the same build/start commands).
3. Ensure a persistent disk is mounted at `/var/data`; SQLite must not live on an ephemeral filesystem.
4. Set `A2A_BASE_URL` to `https://YOUR-SERVICE.onrender.com/a2a/` after Render assigns the hostname.
5. Set `AGENT_BEARER_TOKEN` to the exact token the grader expects, and set an AI provider key.
6. Deploy and test `GET /.well-known/agent-card.json`.
7. Submit the exact `A2A_BASE_URL` value.

## Known hardening checklist before graded checks
This is a starter implementation, not a guarantee of full grader compliance. In particular, test continuation idempotency and atomic cancel/result races, exact A2A error schemas, nested package formats, hidden audit packages, and provider output quality. The continuation implementation must be strengthened to reserve continuation message IDs before processing and atomically bind result nonces to a unique terminal transition before production use. Use only synthetic grader data; never put real invoices or secrets in model prompts.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Bring in the deps manifest plus an OPTIONAL corporate CA bundle. The [t] glob
# makes the CA optional: certs/corp-root-ca.crt is copied if it exists, otherwise
# skipped, while backend/requirements.txt guarantees the COPY always succeeds.
COPY backend/requirements.txt certs/corp-root-ca.cr[t] /tmp/build/

# If a corporate CA is present (e.g. behind a TLS-inspecting proxy), trust it for
# pip (PyPI may be intercepted) first, then append it to certifi AFTER install so
# runtime httpx/openai/requests verify outbound cloud-LLM calls. certifi only
# exists once requirements are installed, so the append must come last.
RUN set -eu; \
    PIP_CERT_ARG=""; \
    if [ -f /tmp/build/corp-root-ca.crt ]; then \
        echo "Trusting corporate CA bundle for pip"; \
        PIP_CERT_ARG="--cert /tmp/build/corp-root-ca.crt"; \
    fi; \
    pip install --no-cache-dir $PIP_CERT_ARG -U pip; \
    pip install --no-cache-dir $PIP_CERT_ARG -r /tmp/build/requirements.txt; \
    if [ -f /tmp/build/corp-root-ca.crt ]; then \
        cat /tmp/build/corp-root-ca.crt >> "$(python -c 'import certifi; print(certifi.where())')"; \
        echo "Appended corporate CA to certifi bundle"; \
    fi

COPY backend/app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

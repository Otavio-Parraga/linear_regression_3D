#!/usr/bin/env bash
# Starts the app locally and exposes it through a Cloudflare quick tunnel.
set -euo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

PORT="${PORT:-8123}"
LOCAL_URL="http://127.0.0.1:${PORT}"
LOG_FILE="$PWD/.cloudflared.log"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
CLOUDFLARED="$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")"

UVICORN_PID=""
TUNNEL_PID=""

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Encerrando..."
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
    [[ -n "$UVICORN_PID" ]] && kill "$UVICORN_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

free_port() {
    # Old tunnels pointing at this port would keep serving a dead URL.
    pkill -f "cloudflared tunnel --url ${LOCAL_URL}" 2>/dev/null || true
    local pids
    pids="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    [[ -z "$pids" ]] && return 0
    echo "Porta ${PORT} ocupada (PIDs: $(echo $pids)); encerrando..."
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 20); do
        lsof -t -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || return 0
        sleep 0.25
    done
    kill -9 $(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null) 2>/dev/null || true
    sleep 0.5
    if lsof -t -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Erro: nao foi possivel liberar a porta ${PORT}." >&2
        exit 1
    fi
}
free_port

echo "Iniciando servidor em ${LOCAL_URL} ..."
"$UV" run uvicorn app.main:app --host 127.0.0.1 --port "$PORT" &
UVICORN_PID=$!

for _ in $(seq 1 60); do
    if curl -fsS -o /dev/null "${LOCAL_URL}/api/model"; then
        break
    fi
    if ! kill -0 "$UVICORN_PID" 2>/dev/null; then
        echo "Erro: uvicorn terminou inesperadamente." >&2
        exit 1
    fi
    sleep 0.5
done
if ! curl -fsS -o /dev/null "${LOCAL_URL}/api/model"; then
    echo "Erro: servidor nao respondeu em ${LOCAL_URL}." >&2
    exit 1
fi

echo "Iniciando tunel Cloudflare (log em ${LOG_FILE}) ..."
: > "$LOG_FILE"
"$CLOUDFLARED" tunnel --url "$LOCAL_URL" --no-autoupdate > "$LOG_FILE" 2>&1 &
TUNNEL_PID=$!

PUBLIC_URL=""
for _ in $(seq 1 60); do
    PUBLIC_URL="$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$LOG_FILE" | head -n1 || true)"
    [[ -n "$PUBLIC_URL" ]] && break
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "Erro: cloudflared terminou inesperadamente. Veja ${LOG_FILE}." >&2
        exit 1
    fi
    sleep 0.5
done

echo
echo "=============================================================="
echo "  Local:    ${LOCAL_URL}"
if [[ -n "$PUBLIC_URL" ]]; then
    echo "  Publico:  ${PUBLIC_URL}"
else
    echo "  Publico:  (URL ainda nao encontrada; veja ${LOG_FILE})"
fi
echo "  Ctrl+C para encerrar."
echo "=============================================================="
echo

wait

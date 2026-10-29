#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu server (run as root from the repository checkout).
# Afterwards: put the keys into /opt/sgm-mm-memory/.env and set the domain in /etc/caddy/Caddyfile.
set -euo pipefail
apt-get update -y && apt-get install -y curl git debian-keyring debian-archive-keyring apt-transport-https
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y && apt-get install -y caddy
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
id memory >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin memory
mkdir -p /opt/sgm-mm-memory && cp -r app scripts requirements.txt .env.example /opt/sgm-mm-memory/
cd /opt/sgm-mm-memory
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
[ -f .env ] || { cp .env.example .env; sed -i "s/^MEMORY_API_KEY=.*/MEMORY_API_KEY=$(openssl rand -hex 24)/" .env; }
sed -i 's#^MEMORY_DATA_DIR=.*#MEMORY_DATA_DIR=/opt/sgm-mm-memory/artifacts#' .env
chmod 600 .env && chown -R memory:memory /opt/sgm-mm-memory
cp "$OLDPWD/deploy/sgm-mm-memory.service" /etc/systemd/system/ && systemctl daemon-reload
systemctl enable --now sgm-mm-memory
[ -f /etc/caddy/Caddyfile.orig ] || cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.orig
cp "$OLDPWD/deploy/Caddyfile" /etc/caddy/Caddyfile
echo "Now: edit /opt/sgm-mm-memory/.env (keys), set the domain in /etc/caddy/Caddyfile, then:"
echo "  systemctl restart sgm-mm-memory caddy && curl https://<domain>/health"

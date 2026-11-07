#!/usr/bin/env bash
# Build the demo and publish it to https://robotcuts.hair (static files served by Caddy).
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")"
bun run build
rsync -az --delete dist/ pujia:/var/www/robotcuts.hair/

#!/usr/bin/env bash
# Builds GNU LibreDWG's dwg2dxf so the CAD-drawing tab can read .dwg files (.dxf works without it).
# Not packaged for Ubuntu 22.04, so it is compiled from the release tarball: ~3 min on Colab. Safe to re-run.
set -euo pipefail

VERSION=0.14
PREFIX=${PREFIX:-/usr/local}

if command -v dwg2dxf >/dev/null 2>&1; then
  echo "dwg2dxf already installed: $(command -v dwg2dxf)"
  exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
echo "==> Downloading LibreDWG $VERSION"
curl -fsSLO "https://ftp.gnu.org/gnu/libredwg/libredwg-$VERSION.tar.xz"
tar xf "libredwg-$VERSION.tar.xz"
cd "libredwg-$VERSION"
echo "==> Building (read-only tools; takes a few minutes)"
./configure --prefix="$PREFIX" --disable-bindings --disable-docs --disable-write >/dev/null
make -j"$(nproc)" >/dev/null
make install >/dev/null
ldconfig 2>/dev/null || true
echo "==> $(dwg2dxf --version 2>&1 | head -1)"

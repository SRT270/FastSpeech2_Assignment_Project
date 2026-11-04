#!/usr/bin/env bash
set -euo pipefail
mkdir -p vendor
if [ ! -d vendor/FastSpeech2/.git ]; then
  git clone https://github.com/ming024/FastSpeech2.git vendor/FastSpeech2
else
  echo "vendor/FastSpeech2 already exists"
fi

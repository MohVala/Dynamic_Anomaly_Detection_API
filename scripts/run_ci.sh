#!/usr/bin/env bash
set -e

export APP_ENV=ci

python -m src.main

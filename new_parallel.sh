#!/usr/bin/env bash
find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 6.4 --early_exit 1

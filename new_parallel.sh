#!/usr/bin/env bash
find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 0 -n 9.8 -s 0.2 --early_exit 1 -t 1 &
find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 0 -n 9.8 -s 0.2 --early_exit 1 -t 2 &
find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 0 -n 9.8 -s 0.2 --early_exit 1 -t 3 &
find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 0 -n 9.8 -s 0.2 --early_exit 1 -t 4 &

find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 10.0 -n 150.0 -s 5 --early_exit 1 -t 1 &
find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 10.0 -n 150.0 -s 5 --early_exit 1 -t 2 &
find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 10.0 -n 150.0 -s 5 --early_exit 1 -t 3 &
find ./experiments -name "*.json" | parallel ./build/evaluate  -e {} -r 0 -p 0 -f  1 -m 10.0 -n 150.0 -s 5 --early_exit 1 -t 4 &

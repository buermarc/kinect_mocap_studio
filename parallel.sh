#!/usr/bin/env bash
find ./experiments -name "*.json" | parallel EXPERIMENT_JSON={} sh ./start_evaluate_grid.sh

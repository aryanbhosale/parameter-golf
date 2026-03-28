#!/bin/bash
# Parameter Golf v3 — Go for #1
set -e

SCRIPT_DIR="records/track_10min_16mb/2026-03-25_11L_ParallelMuon_MLP3x_TTT"

# ===== ULTIMATE CONFIG: Every validated technique stacked =====
# slope 0.75 + VE128 + no-gated-attn + no-SWA + QAT-50% + mHC + LZMA
# + Muon TTT + entropy-adaptive epochs + per-layer LR + momentum 0.95
# + LR 0.027 + warmdown 3700
run_ultimate() {
    local SEED=$1
    echo "=== ULTIMATE, seed=$SEED ==="
    cp "$SCRIPT_DIR/train_gpt_ultimate.py" train_gpt.py

    SEED=$SEED MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
    USE_EMA=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
    TRAIN_BATCH_TOKENS=786432 \
    USE_TTT=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
    TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.95 TTT_BATCH_SEQS=32 \
    TTT_MUON=1 TTT_NS_STEPS=3 TTT_ENTROPY_ADAPT=1 \
    USE_LZMA=1 \
    torchrun --nproc_per_node=8 --standalone train_gpt.py \
        > run_ultimate_seed${SEED}.log 2>&1

    echo "ULTIMATE seed=$SEED done. Check run_ultimate_seed${SEED}.log"
}

# ===== QUICK 1xH100 SCREENING =====
screen_ultimate() {
    echo "=== Screening ULTIMATE on 1xH100 ==="
    cp "$SCRIPT_DIR/train_gpt_ultimate.py" train_gpt.py

    PYTHONUNBUFFERED=1 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=256 \
    USE_EMA=0 SWA_ENABLED=0 WARMDOWN_ITERS=300 \
    TRAIN_BATCH_TOKENS=524288 USE_TTT=0 USE_LZMA=1 \
    python -u train_gpt.py > screen_ultimate.log 2>&1

    echo "Screening done. Check screen_ultimate.log"
}

# ===== USAGE =====
echo "Parameter Golf v3 — Go for #1"
echo ""
echo "  1. Screen on 1xH100 (check compile + artifact size):"
echo "     bash run_experiments.sh screen"
echo ""
echo "  2. Run 3-seed on 8xH100:"
echo "     bash run_experiments.sh run 1337"
echo "     bash run_experiments.sh run 42"
echo "     bash run_experiments.sh run 2024"
echo ""

case "${1:-help}" in
    run)    run_ultimate "${2:-1337}" ;;
    screen) screen_ultimate ;;
    *)      echo "Usage: bash run_experiments.sh {run|screen} [seed]" ;;
esac

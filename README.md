# SAND ViT Baseline (Phase 1) — README

## Overview
This repository implements the Phase-1 baseline for the SAND challenge:
- A simple ViT (Vision Transformer) classifier pipeline that **verifies audio files load and flow** through the model.
- No MFCC or advanced audio preprocessing is used in Phase 1 — a minimal waveform → image-like transform is applied so the ViT can accept inputs.
- The code reproduces the baseline subject-level training/validation split using the sheets from `sand_task_1.xlsx` (Training Baseline / Validation Baseline).

## File layout expected
Place your `TASK1/` folder somewhere (e.g. `~/Desktop/TASK1`) with this structure:
x
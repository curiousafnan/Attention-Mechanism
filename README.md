# Day 39 — Attention Mechanism

Luong multiplicative attention over an LSTM encoder-decoder, built
from scratch with raw nn.Parameter tensors.

## Core Classes
Encoder    — returns all hidden states, no output projection
Attention  — score → softmax → blend, one (hidden, hidden) matrix
Decoder    — attention inside the timestep loop, cat(h, context)
Seq2SeqAttention — wrapper, fit / predict / evaluate_autoregressive

## Task
Sequence reversal. [6,18,3,12,7,9] → [9,7,12,3,18,6]
Correct alignment known in advance, so the heatmap verifies the
implementation rather than merely illustrating it.

## Results (5 epochs)
| Metric                        | Value |
|-------------------------------|-------|
| Autoregressive token accuracy | 1.000 |
| Autoregressive sequence acc   | 1.000 |

Alignment: output step 0 → source 5, step 1 → source 4, … step 5 → 0.
Clean anti-diagonal, weights near 1.0/0.0.

## Key Insight
The context vector is rebuilt at every decoder step instead of being
fixed at h_final. Nobody labelled the alignment — it emerged because
attending correctly lowers the loss.

aW learns a reusable scoring rule, not a position mapping. Where one
rule is not enough, you need several in parallel — multi-head attention.

## Run
python demo.py
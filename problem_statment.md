# Day 39 — Attention Mechanism

## Problem
In seq2seq, the encoder reads the whole source and passes only its
final hidden state to the decoder. Every intermediate state is
discarded. A 5-token sequence and a 500-token sequence both get the
same fixed-size vector, so the decoder's view of the source cannot
grow with the input.

Sutskever's input-reversal trick eased this — it brought the first
source tokens closer to where the decoder starts — but it only
redistributed distance. The bottleneck itself remained.

## Core Questions
- What does the decoder lose when h₁..h₅ are discarded?
- How should it decide which source positions matter at each step?
- With no loss of its own, how does the encoder learn?
- What distinguishes a model that attends correctly from one that
  memorizes?

## Requirements
- Encoder returning all hidden states, not just the final one
- Attention module computing a fresh context vector per decoder step
- Decoder consuming that context when predicting
- Sequence-reversal task, where correct alignment is known in advance
- Alignment heatmap as the verification
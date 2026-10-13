# Approach — Day 39

## The Three Steps
At every decoder step t:

1. SCORE — compare the decoder state s against each encoder state
       e_i = (s @ aW) · h_i          for i = 1..src_len
   Produces src_len unbounded numbers.

2. SOFTMAX — turn scores into weights
       α = softmax(e)
   Positive, summing to 1. A distribution over source positions.

3. BLEND — weighted sum of encoder states
       c = Σ αᵢ hᵢ
   Shape (batch, hidden). A context vector built for this step.

In plain seq2seq, c was h_final forever. Here it is rebuilt every
step, so the decoder looks at different source positions as it goes.

## Scoring Variant
Luong multiplicative: score(s, h) = s @ aW · h

aW is (hidden, hidden) — a learned bridge. Encoder and decoder are
separate LSTMs with separate weights, so their hidden states do not
live in a directly comparable space. aW maps s into the encoder's
space before the dot product.

| Variant        | Score                      | Parameters |
|----------------|----------------------------|------------|
| Dot            | sᵀh                        | none       |
| Multiplicative | sᵀWh                       | one matrix |
| Additive       | vᵀtanh(W_s s + W_h h)      | two + vec  |

Multiplicative chosen: fewer parameters, one matmul instead of a
tanh network, LSTM cell untouched, and it is the direct ancestor of
scaled dot-product attention in transformers.

## Architecture
Four separate classes, no circular dependencies:

  Encoder    — LSTM, 12 gate parameters, NO output projection
               (it predicts nothing; it only reads)
               returns all_h (batch, src_len, hidden), h, c

  Attention  — aW only. forward(s, all_h) → context, alpha

  Decoder    — LSTM + its own Attention instance. Attention runs
               INSIDE the timestep loop, after the LSTM step, using
               the current h. dW_hy is (2*hidden, vocab) because
               h and context are concatenated before projection.

  Seq2SeqAttention — owns both, holds fit/predict/evaluate.
               One optimizer over all parameters.

## Luong vs Bahdanau Placement
Luong (used here): LSTM step first, then attention on the resulting
h, then concatenate for the prediction. Context influences what word
comes out, not how the hidden state evolves.

Bahdanau: attention on h_{t-1} first, context fed into the gates as
extra input. Context shapes the recurrence itself, at the cost of
every gate's W growing to (embedding_dim + hidden, hidden).

## Task: Sequence Reversal
    src     = [6, 18, 3, 12, 7, 9]
    tgt_in  = [START, 9, 7, 12, 3, 18, 6]
    tgt_out = [9, 7, 12, 3, 18, 6, END]

Chosen over translation deliberately. The correct alignment is known
in advance — a clean anti-diagonal — so the heatmap becomes a real
test of correctness rather than a vibe check. Translation would have
spent the session on data plumbing and produced an alignment nobody
could verify by eye.

Vocabulary: 0=PAD, 1=START, 2=END, 3-20 = the digits.

## Config
src_vocab 21 | tgt_vocab 21 | embedding 64 | hidden 128
50,000 samples | seq_len 6 | batch 32 | Adam lr=0.001 | 5 epochs
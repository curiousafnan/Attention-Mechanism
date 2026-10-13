# Learnings — Day 39

## The Bottleneck, and Why Reversal Was Only a Trick
Without reversal, the decoder's first output needs the source's first
token — which sat furthest back in the encoder's pass and decayed
most. Reversing the source brings those first tokens to the end of
the encoder's reading, so they are freshest when the decoder starts.

But the total distance is unchanged. Reversal helps the beginning of
the sentence by hurting the end. Sutskever's own paper states they
had no complete explanation for why it worked as well as it did.

The bottleneck — one fixed vector for any input length — was still
there. Attention is what removes it.

## Attention in Three Steps
    score   → src_len unbounded numbers, from s @ aW dotted with each h_i
    softmax → the same numbers as a distribution, summing to 1
    blend   → Σ αᵢ hᵢ, a context vector for this step only

Order matters in the description: aW produces SCORES; softmax turns
scores into ALPHAS. Not the reverse.

## The Encoder Learns Without a Loss
It has no output projection and is never compared against a target.
It learns because it sits in the same computation graph:

    loss → decoder projection → context → α → all_h → encoder gates

Every operation between is differentiable, so loss.backward() reaches
the encoder. It learns to produce hidden states that are worth
attending to — if h₃ poorly represents source position 3, no amount
of correct weighting helps, and the gates get corrected.

Removing eW_hy was correct for the same reason: nothing in the loss
touches it, so its gradient would always be zero. Dead weight.

## Why the Heatmap Proves What Accuracy Cannot
100% accuracy says the model produces the right output. It does not
say how. An LSTM large enough could memorize reversal patterns and
leave the attention weights flat at ~0.17 each — unused.

The heatmap shows a sharp anti-diagonal with weights near 1.0 and
0.0. Output step 0 attends to source position 5, step 1 to position
4, and so on down to 0. The model is pointing at exactly one source
position per step.

And nothing in the training data said to. There are no alignment
labels — only the reversed sequence. The anti-diagonal emerged
because attending there minimizes loss. Same shape as Word2Vec:
train on one objective, get a different useful structure as a
byproduct.

## What aW Actually Learns
Not "position 5 maps to position 0" — that would break the moment
sequence length changed.

It learns a reusable scoring rule: how to compare a decoder state
against an encoder state to judge relevance. One geometric
relationship that holds across every sequence, the same way W_hh in
an RNN or a conv filter is one rule applied at every position.

Where one matrix stops being enough: when a task needs several
unrelated alignments simultaneously — subjects to verbs AND
adjectives to nouns — a single aW must average across them.
The fix is several attention mechanisms in parallel, each with its
own projection. That is multi-head attention, Day 42.

## Results
    5 epochs, 50,000 samples, seq_len 6

    Teacher-forced accuracy:        1.0000
    Autoregressive token accuracy:  1.0000
    Autoregressive sequence acc:    1.0000

    Source:    [6, 18, 3, 12, 7, 9]
    Predicted: [9, 7, 12, 3, 18, 6, 2]   ← exact

Autoregressive matters more than teacher-forced: the model generates
from its own outputs with no help from the correct answer.

Compare to Day 37, where the RNN needed 1000 epochs to produce
broken English.

Output step 6 attends to position 0 — that is the END token, with
nothing left to copy, parking on the last position used. Expected.

## Unverified
Everything here was trained and evaluated at seq_len 6. Whether the
learned scoring rule generalizes to longer sequences was not tested.

## Bugs Fixed
- Attention placed AFTER the decoder loop instead of inside it —
  that is plain seq2seq again, one context for the whole output
- Passed h0 (constant) to attention instead of the current h
- Encoder parameter names (eW_*) used inside the Decoder — copy-paste
- torch.softmax called without dim (required, not optional)
- Returned the unsqueezed (batch, src_len, 1) alpha instead of
  (batch, src_len)
- tried reshaping scores before stacking; the list holds (batch,)
  tensors, stacked at dim=-1 to give (batch, src_len)
- Missing super().__init__() in the wrapper — would have left
  model.parameters() empty and training silently doing nothing
- fit() epoch bookkeeping indented inside the batch loop
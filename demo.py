from src.seq2seq import *

generator = DataGenerator()
x, y_in, y_out = generator.generate_pair(n_samples=50000, seq_len=6)
x, y_in, y_out = torch.tensor(x),torch.tensor(y_in),torch.tensor(y_out)

x_train_len = int(
    len(x) * 0.8
)

x_train = x[:x_train_len]
y_train_in = y_in[:x_train_len]
y_train_out = y_out[:x_train_len]

x_test = x[x_train_len:]
y_test_in = y_in[x_train_len:]
y_test_out = y_out[x_train_len:]

train_dataset = CustomDataset(
    x_train,
    y_train_in,
    y_train_out
)

test_dataset = CustomDataset(
    x_test,
    y_test_in,
    y_test_out
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False
)

import matplotlib.pyplot as plt
import torch

def plot_attention_heatmap(model, src_sample, start_token=1, end_token=2, max_len=10, figsize=(8, 6)):
    model.eval()

    device = next(model.parameters()).device

    if not isinstance(src_sample, torch.Tensor):
        src_sample = torch.tensor(src_sample, dtype=torch.long)

    src_sample = src_sample.to(device)

    if src_sample.ndim == 1:
        src_sample = src_sample.unsqueeze(0)

    preds, attn = model.predict(
        src_sample,
        start_token=start_token,
        end_token=end_token,
        max_len=max_len
    )

    src_tokens = src_sample[0].detach().cpu().tolist()
    pred_tokens = preds[0].detach().cpu().tolist()
    attn_matrix = attn[0].detach().cpu()   # (pred_len, src_len)

    plt.figure(figsize=figsize)
    plt.imshow(attn_matrix, aspect="auto")
    plt.colorbar(label="Attention Weight")

    plt.xticks(range(len(src_tokens)), src_tokens)
    plt.yticks(range(len(pred_tokens)), pred_tokens)

    plt.xlabel("Source Positions / Source Tokens")
    plt.ylabel("Generated Output Tokens")
    plt.title("Attention Alignment Heatmap")

    plt.tight_layout()
    plt.show()

    return pred_tokens, attn_matrix

def inspect_attention_alignment(attn_matrix):
    # attn_matrix shape: (tgt_len, src_len)
    best_src_positions = attn_matrix.argmax(dim=1).tolist()
    print("Most attended source position at each output step:")
    for t, pos in enumerate(best_src_positions):
        print(f"Output step {t} -> source position {pos}")

seq2seq = Seq2SeqAttention(src_vocab=21, tgt_vocab=21,embedding_dim=64,hidden_size=128)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=0.001)
seq2seq.fit(dataloader=train_loader,epochs=5,loss_fn=loss_fn,optimizer=optimizer)
token_acc, seq_acc = seq2seq.evaluate_autoregressive(test_loader)
print(f"Autoregressive Token Accuracy: {token_acc:.4f}")
print(f"Autoregressive Sequence Accuracy: {seq_acc:.4f}")

sample_src, sample_tgt_in, sample_tgt_out = test_dataset[0]

pred_tokens, attn_matrix = plot_attention_heatmap(
    seq2seq,
    sample_src,
    start_token=1,
    end_token=2,
    max_len=len(sample_tgt_out)
)

print("Source:   ", sample_src.tolist())
print("Target:   ", sample_tgt_out.tolist())
print("Predicted:", pred_tokens)
inspect_attention_alignment(attn_matrix)
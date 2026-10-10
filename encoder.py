import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Encoder(nn.Module):
    def __init__(self,vocab_size:int, embedding_dim:int, hidden_size:int,):
        super().__init__()
        if min(vocab_size, embedding_dim,hidden_size)<=0:
            raise ValueError("All sizes must be positive.")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        # Encoder
        self.eW_fh = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.eW_fhh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.eb_fh = nn.Parameter(torch.zeros(hidden_size))
        self.eW_ih = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.eW_ihh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.eb_ih = nn.Parameter(torch.zeros(hidden_size))
        self.eW_cih = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.eW_cihh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.eb_cih = nn.Parameter(torch.zeros(hidden_size))
        self.eW_oh = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.eW_ohh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.eb_oh = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_uniform_(self.eW_fh)
        nn.init.xavier_uniform_(self.eW_ih)
        nn.init.xavier_uniform_(self.eW_cih)
        nn.init.xavier_uniform_(self.eW_oh)
        nn.init.orthogonal_(self.eW_fhh)
        nn.init.orthogonal_(self.eW_ihh)
        nn.init.orthogonal_(self.eW_cihh)
        nn.init.orthogonal_(self.eW_ohh)


    def forward(self,x: torch.Tensor,h0: torch.Tensor | None = None,c0: torch.Tensor | None = None,) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(
                "x must have shape (batch_size, seq_len)."
            )

        if x.dtype != torch.long:
            raise TypeError(
                "x must contain character IDs as torch.long."
            )

        batch_size, seq_len = x.shape
        if seq_len == 0:
            raise ValueError("Input sequences cannot be empty.")
        embedded = self.embedding(x)

        if h0 is None:
            h = embedded.new_zeros((batch_size, self.hidden_size))
        else:
            if h0.shape != (batch_size, self.hidden_size):
                raise ValueError(
                    "h0 must have shape (batch_size, hidden_size)."
                )

            if (
                h0.device != embedded.device
                or h0.dtype != embedded.dtype
            ):
                raise ValueError(
                    "h0 must match the embedding device and dtype."
                )

            h = h0

        if c0 is None:
            c = embedded.new_zeros((batch_size, self.hidden_size))
        else:
            if c0.shape != (batch_size, self.hidden_size):
                raise ValueError(
                    "c0 must have shape (batch_size, hidden_size)."
                )

            if (
                c0.device != embedded.device
                or c0.dtype != embedded.dtype
            ):
                raise ValueError(
                    "c0 must match the embedding device and dtype."
                )

            c = c0

        hidden_states = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            ft = torch.sigmoid(
                x_t @ self.eW_fh
                + h @ self.eW_fhh
                + self.eb_fh
            )
            it = torch.sigmoid(
                x_t @ self.eW_ih
                + h @ self.eW_ihh
                + self.eb_ih
            )
            cit = torch.tanh(
                x_t @ self.eW_cih
                + h @ self.eW_cihh
                + self.eb_cih
            )
            ot = torch.sigmoid(
                            x_t @ self.eW_oh
                            + h @ self.eW_ohh
                            + self.eb_oh
                        )
            
            c = c * ft + cit * it
            h = torch.tanh(c) * ot

            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states, dim=1)
        return hidden_states, h,c

class Attention(nn.Module):
    def __init__(self, hidden_size:int,):
        super().__init__()
        self.hidden_size = hidden_size
        self.aW = nn.Parameter(torch.empty(hidden_size,hidden_size))
        nn.init.xavier_uniform_(self.aW)

    def forward(self,s,all_h):
        pre_e = s @ self.aW
        _, src_len, _ = all_h.shape
        scores = []
        for i in range(src_len):
            h_i = all_h[:,i,:]
            e_i = pre_e * h_i
            e_i = torch.sum(e_i,dim=1)
            scores.append(e_i)
        alpha = torch.softmax(torch.stack(scores, dim=-1), dim=-1)
        c = (alpha.unsqueeze(2) * all_h).sum(dim=1)
        return c, alpha  


class Decoder(nn.Module):
    def __init__(self,vocab_size:int,embedding_dim:int, hidden_size:int,):
        super().__init__()
        if min(vocab_size, embedding_dim,hidden_size)<=0:
            raise ValueError("All sizes must be positive.")
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.attention = Attention(hidden_size)
        self.hidden_size = hidden_size
        self.dW_fh = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.dW_fhh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.db_fh = nn.Parameter(torch.zeros(hidden_size))
        self.dW_ih = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.dW_ihh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.db_ih = nn.Parameter(torch.zeros(hidden_size))
        self.dW_cih = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.dW_cihh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.db_cih = nn.Parameter(torch.zeros(hidden_size))
        self.dW_oh = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.dW_ohh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.db_oh = nn.Parameter(torch.zeros(hidden_size))
        self.dW_hy = nn.Parameter(torch.empty(2*hidden_size,vocab_size))
        self.db_y = nn.Parameter(torch.zeros(vocab_size))
        nn.init.xavier_uniform_(self.dW_fh)
        nn.init.xavier_uniform_(self.dW_ih)
        nn.init.xavier_uniform_(self.dW_cih)
        nn.init.xavier_uniform_(self.dW_oh)
        nn.init.orthogonal_(self.dW_fhh)
        nn.init.orthogonal_(self.dW_ihh)
        nn.init.orthogonal_(self.dW_cihh)
        nn.init.orthogonal_(self.dW_ohh)
        nn.init.xavier_uniform_(self.dW_hy)

    def forward(self,x: torch.Tensor,h_all:torch.Tensor,h0: torch.Tensor | None = None,c0: torch.Tensor | None = None,) -> tuple[torch.Tensor, torch.Tensor]:
            if x.ndim != 2:
                raise ValueError(
                    "x must have shape (batch_size, seq_len)."
                )
    
            if x.dtype != torch.long:
                raise TypeError(
                    "x must contain character IDs as torch.long."
                )
    
            batch_size, seq_len = x.shape
            if seq_len == 0:
                raise ValueError("Input sequences cannot be empty.")
            embedded = self.embedding(x)
    
            if h0 is None:
                h = embedded.new_zeros((batch_size, self.hidden_size))
            else:
                if h0.shape != (batch_size, self.hidden_size):
                    raise ValueError(
                        "h0 must have shape (batch_size, hidden_size)."
                    )
    
                if (
                    h0.device != embedded.device
                    or h0.dtype != embedded.dtype
                ):
                    raise ValueError(
                        "h0 must match the embedding device and dtype."
                    )
    
                h = h0
    
            if c0 is None:
                c = embedded.new_zeros((batch_size, self.hidden_size))
            else:
                if c0.shape != (batch_size, self.hidden_size):
                    raise ValueError(
                        "c0 must have shape (batch_size, hidden_size)."
                    )
    
                if (
                    c0.device != embedded.device
                    or c0.dtype != embedded.dtype
                ):
                    raise ValueError(
                        "c0 must match the embedding device and dtype."
                    )
    
                c = c0
    
            logits_list = []
            alphas_list = []

            for t in range(seq_len):
                x_t = embedded[:, t, :]
                ft = torch.sigmoid(
                    x_t @ self.dW_fh
                    + h @ self.dW_fhh
                    + self.db_fh
                )
                it = torch.sigmoid(
                    x_t @ self.dW_ih
                    + h @ self.dW_ihh
                    + self.db_ih
                )
                cit = torch.tanh(
                    x_t @ self.dW_cih
                    + h @ self.dW_cihh
                    + self.db_cih
                )
                ot = torch.sigmoid(
                                x_t @ self.dW_oh
                                + h @ self.dW_ohh
                                + self.db_oh
                            )
                
                c = c * ft + cit * it
                h = torch.tanh(c) * ot

                contx,alpha = self.attention.forward(s=h,all_h=h_all)
                combined = torch.cat((h,contx),dim=-1)
                logit_t = combined @ self.dW_hy + self.db_y
                logits_list.append(logit_t)
                alphas_list.append(alpha)

            logits = torch.stack(logits_list, dim=1)    # (batch, tgt_len, vocab)
            alphas = torch.stack(alphas_list, dim=1)    # (batch, tgt_len, src_len)
            return logits, alphas


class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim, hidden_size):
        super().__init__()
        self.encoder = Encoder(src_vocab, embedding_dim, hidden_size)
        self.decoder = Decoder(tgt_vocab, embedding_dim, hidden_size)

    def forward(self, src, tgt):
        all_h, h, c = self.encoder(src)
        logits, alphas = self.decoder(tgt, all_h, h, c)
        return logits, alphas

    def fit()
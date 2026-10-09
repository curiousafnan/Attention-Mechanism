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
        e = torch.softmax(torch.stack(scores, dim=1))
        return e        


class Decoder(nn.Module):
    def __init__(self,vocab_size:int, embedding_dim:int, hidden_size:int,):
        super().__init__()
        if min(vocab_size, embedding_dim,hidden_size)<=0:
            raise ValueError("All sizes must be positive.")
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
        self.dW_hy = nn.Parameter(torch.empty(hidden_size,vocab_size))
        self.db_y = nn.Parameter(torch.zeros(vocab_size))
        self.aW = nn.Parameter(torch.empty(hidden_size,hidden_size))
        nn.init.xavier_uniform_(self.dW_fh)
        nn.init.xavier_uniform_(self.dW_ih)
        nn.init.xavier_uniform_(self.dW_cih)
        nn.init.xavier_uniform_(self.dW_oh)
        nn.init.orthogonal_(self.dW_fhh)
        nn.init.orthogonal_(self.dW_ihh)
        nn.init.orthogonal_(self.dW_cihh)
        nn.init.orthogonal_(self.dW_ohh)
        nn.init.xavier_uniform_(self.dW_hy)
        nn.init.xavier_uniform_(self.aW)
import numpy as np
import torch
import torch.nn as nn


def activation_from_str(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "tanh":
        return nn.Tanh()
    if n == "logistic" or n == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name!r}")


class TorchMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        activation: nn.Module,
        dropout_p: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPModel:
    """Binary MLP in PyTorch: uses CUDA when available, otherwise CPU."""

    def __init__(self, params: dict):
        self.params = dict(params)
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if self.torch_device.type == "cuda" else "cpu"
        self.net: TorchMLP | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        hidden = tuple(self.params.get("hidden_layer_sizes", (256, 128)))
        act_name = str(self.params.get("activation", "relu"))
        dropout_p = float(self.params.get("dropout", 0.2))
        lr = float(self.params.get("learning_rate_init", 1e-3))
        wd = float(self.params.get("alpha", 0.0))
        bs = int(self.params.get("batch_size", 256))
        max_epochs = int(self.params.get("max_iter", 200))
        val_frac = float(self.params.get("validation_fraction", 0.1))
        early_stop = bool(self.params.get("early_stopping", False))
        patience = int(self.params.get("n_iter_no_change", 10))
        seed = int(self.params.get("random_state", 42))

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        n = X.shape[0]
        if n < 2:
            raise ValueError("Need at least 2 samples to train MLPModel.")

        n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
        if n_val >= n:
            n_val = max(1, n // 10)

        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        self.mu = X_train.mean(axis=0, keepdims=True)
        self.sigma = X_train.std(axis=0, keepdims=True)
        self.sigma[self.sigma == 0] = 1.0
        X_train = (X_train - self.mu) / self.sigma
        X_val = (X_val - self.mu) / self.sigma

        act = activation_from_str(act_name)
        self.net = TorchMLP(X_train.shape[1], hidden, act, dropout_p).to(self.torch_device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)

        Xt = torch.tensor(X_train, device=self.torch_device)
        yt = torch.tensor(y_train, device=self.torch_device)
        Xv = torch.tensor(X_val, device=self.torch_device)
        yv = torch.tensor(y_val, device=self.torch_device)

        best_val = float("inf")
        best_state: dict | None = None
        no_improve = 0

        for _ in range(max_epochs):
            self.net.train()
            idx = torch.randperm(Xt.shape[0], device=self.torch_device)
            for i in range(0, Xt.shape[0], bs):
                b = idx[i : i + bs]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(self.net(Xt[b]), yt[b])
                loss.backward()
                opt.step()

            if early_stop and n_val > 0:
                self.net.eval()
                with torch.no_grad():
                    v_loss = float(nn.functional.binary_cross_entropy_with_logits(self.net(Xv), yv).item())
                if v_loss < best_val - 1e-6:
                    best_val = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        if early_stop and best_state is not None:
            self.net.load_state_dict(best_state)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None or self.mu is None or self.sigma is None:
            raise RuntimeError("Model is not fitted.")
        Xn = (np.asarray(X, dtype=np.float32) - self.mu) / self.sigma
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(Xn, device=self.torch_device))
            return torch.sigmoid(logits).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

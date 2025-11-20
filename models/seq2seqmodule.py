import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Seq2SeqForecastingModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        num_autoregressive_steps: int = 1,
        teacher_forcing_ratio: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = float(lr)
        self.num_autoregressive_steps = num_autoregressive_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, past: torch.Tensor, future_len: int) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError

    def _shared_step(self, batch):
        past, future = batch
        preds = self.forward(past, future.size(-1))
        loss = F.mse_loss(preds, future)
        return loss, preds

    def training_step(self, batch, batch_idx):
        if self.num_autoregressive_steps > 1:
            loss = self._training_step_autoregressive(batch, batch_idx)
        else:
            loss = self._training_step_single(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _training_step_single(self, batch, batch_idx):
        """Standard single-step training (current behavior)."""
        loss, _ = self._shared_step(batch)
        return loss

    def _training_step_autoregressive(self, batch, batch_idx):
        """Autoregressive training with teacher forcing.

        Each batch contains:
            past: [batch, state_dim, input_len]
            future: [batch, state_dim, target_len]

        This trains the model to handle autoregressive prediction by:
        1. Making predictions from the input window (predicting target_len timesteps)
        2. Using either ground truth (teacher forcing) or predictions for next input
        3. Computing loss at each autoregressive step

        Note: The future tensor should be large enough to contain num_autoregressive_steps
        worth of data (or the same size, in which case we step through it).
        """
        past, future = batch
        state_dim = past.size(1)
        input_len = past.size(-1)
        target_len = future.size(-1)

        total_loss = torch.tensor(0.0, device=past.device)
        history = past.clone()

        for step in range(self.num_autoregressive_steps):
            # Take last input_len timesteps as input window
            input_window = history[:, :, -input_len:]

            # Predict next target_len timesteps (fixed output length)
            pred_step = self.forward(input_window, target_len)

            # For loss computation, use the same ground truth (target_len timesteps)
            # This is appropriate when the dataset provides target_len as the horizon
            step_loss = F.mse_loss(pred_step, future)
            total_loss = total_loss + step_loss

            # Decide: use ground truth or prediction for next input (teacher forcing)
            use_ground_truth = torch.rand(1).item() < self.teacher_forcing_ratio
            if use_ground_truth:
                # Use actual ground truth for next window
                next_input = future
            else:
                # Use model prediction for next window
                next_input = pred_step

            # Update history by appending next input
            history = torch.cat([history, next_input], dim=-1)

        # Average loss over steps
        return total_loss / self.num_autoregressive_steps

    def validation_step(self, batch, batch_idx):
        loss, preds = self._shared_step(batch)
        mae = F.l1_loss(preds, batch[1])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)

    def on_fit_start(self):
        # Push all saved hparams to MLflow (requires child to call save_hyperparameters)
        if getattr(self, "logger", None) and hasattr(self.logger, "log_hyperparams"):
            try:
                self.logger.log_hyperparams(dict(self.hparams)) # type: ignore
            except Exception:
                pass
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
import lightning as L
import torch
import torch.nn as nn
from typing import Optional


class VisionEncoderDecoderModel(L.LightningModule):

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        tokenizer=None,
        vocab_json_path=None,
        dataset_name=None,
        init_from_checkpoint_path: Optional[str] = None,
        max_sequence_length=MAX_LEN,
        use_beam_search=False,
        optimizer: L.OptimizerCallable = torch.optim.AdamW,
        scheduler: L.LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        device=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._device = device or encoder.device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        # assert self.encoder.device == self._device

    def forward(self, pixel_values=None, input_ids=None, return_loss=False, encoder_hidden_states=None):
        if encoder_hidden_states is None:
            image_embedidngs = self.encoder(pixel_values=pixel_values).last_hidden_state
        else:
            image_embedidngs = encoder_hidden_states
        out = self.decoder(input_ids, encoder_hidden_states=image_embedidngs, return_loss=return_loss)
        return out

    def training_step(self, batch_dict, batch_idx):
        pixel_values = batch_dict["pixel_values"].to(self._device)
        input_ids = batch_dict["input_ids"].to(self._device)

        loss = self(pixel_values=pixel_values, input_ids=input_ids, return_loss=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch_dict):
        pixel_values = batch_dict["pixel_values"].to(self._device)
        input_ids = batch_dict["input_ids"].to(self._device)

        loss = self(pixel_values=pixel_values, input_ids=input_ids, return_loss=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=2e-4)

import lightning as L
import torch
import torch.nn as nn
from typing import Optional, List
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

from MEGABYTE_pytorch.megabyte import MEGABYTE


class VisionEncoderDecoderModel(L.LightningModule):

    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        tokenizer=None,
        vocab_json_path=None,
        dataset_name=None,
        init_from_checkpoint_path: Optional[str] = None,
        use_beam_search=False,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        device=None,
        lr=2e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._device = device or encoder.device
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_class = optimizer
        self.schduler_class = scheduler
        self.lr = lr
        if self.encoder.config.hidden_size != self.decoder.hidden_sizes[0]:
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.hidden_sizes[0])
        else:
            self.enc_to_dec_proj = None

    def forward(self, pixel_values=None, input_ids=None, return_loss=False, encoder_hidden_states=None):
        if encoder_hidden_states is None:
            image_embeddings = self.encoder(pixel_values=pixel_values).last_hidden_state
        else:
            image_embeddings = encoder_hidden_states
        if self.enc_to_dec_proj is not None:
            image_embeddings = self.enc_to_dec_proj(image_embeddings)
        out = self.decoder(input_ids, encoder_hidden_states=image_embeddings, return_loss=return_loss)
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
        optimizer = self.optimizer_class(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        scheduler = self.schduler_class(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "metric": "val_loss"}


class Megabyte(L.LightningModule):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_sizes: List[int],
        num_hidden_layers: List[int],
        max_sequence_lengths: List[int],
        dim_head: int = 64,
        num_heads: int = 8,
        attention_dropout_prob: float = 0.1,
        feed_forward_scaleup: int = 4,
        feed_forward_dropout_prob: float = 0.1,
        pad_token_id: int = 0,
        eos_token_id: int = None,
        bos_token_id: int = None,
        rel_pos: bool = False,
        pos_emb: bool = False,
        flash_attn: bool = False,
        add_cross_attention: bool = False,
        lr=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MEGABYTE(
            vocab_size=vocab_size,
            hidden_sizes=hidden_sizes,
            num_hidden_layers=num_hidden_layers,
            max_sequence_lengths=max_sequence_lengths,
            dim_head=dim_head,
            num_heads=num_heads,
            attention_dropout_prob=attention_dropout_prob,
            feed_forward_scaleup=feed_forward_scaleup,
            feed_forward_dropout_prob=feed_forward_dropout_prob,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            rel_pos=rel_pos,
            pos_emb=pos_emb,
            flash_attn=flash_attn,
            add_cross_attention=add_cross_attention,
        )
        self.lr = lr

    def forward(self, input_ids, return_loss=False):
        return self.model(input_ids, return_loss=return_loss)

    def training_step(self, batch_dict, batch_idx=None):
        loss = self(batch_dict["input_ids"], return_loss=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch_dict, batch_idx=None):
        loss = self(batch_dict["input_ids"], return_loss=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch_dict, batch_idx=None):
        loss = self(batch_dict["input_ids"], return_loss=True)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr, betas=(0.9, 0.98), weight_decay=0.1)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, min_lr=1e-5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": sched, "monitor": "val_loss"}

import os
import glob
import click
import logging

import comet_ml

from dataset import get_dataset_by_name, TokenizerDataModule
import IPython
import pdb

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import pandas as pd

# from aim.pytorch_lightning import AimLogger

logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# this hides a warning thrown by huggingface transformers
# https://github.com/huggingface/transformers/issues/5486
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"  #  set to false is processes stuck


class LMForSequenceClassification(pl.LightningModule):
    def __init__(
        self,
        src_model: str,
        learning_rate: float,
        regularization: str = None,
        reg_strength: float = 0.01,
        weight_decay: float = 0.0,
        warmup_train_perc: float = None,
        train_steps_count: int = None,
        class_weights: torch.Tensor = None
    ):
        super().__init__()

        if regularization and regularization == "norm":
            # use custom transformers from:
            # https://github.com/gorokoba560/norm-analysis-of-transformer
            # the norm evaluation is currently supported on Bert only
            import transformers
            from transformers import BertForSequenceClassification

            assert transformers.__version__ == "3.0.0"
            self.model = BertForSequenceClassification.from_pretrained(src_model)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(src_model)

        self.save_hyperparameters()
        
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)

        #  metrics
        self.train_acc = pl.metrics.Accuracy()
        self.train_F1 = pl.metrics.F1(num_classes=2, average="macro")
        self.val_acc = pl.metrics.Accuracy()
        self.val_F1 = pl.metrics.F1(num_classes=2, average="macro")
        self.test_acc = pl.metrics.Accuracy()
        self.test_F1 = pl.metrics.F1(num_classes=2, average="macro")
        self.test_prec = pl.metrics.Precision(num_classes=2, average="macro")
        self.test_rec = pl.metrics.Recall(num_classes=2, average="macro")

    def forward(self, **inputs):
        return self.model(**inputs)

    def forward_pass(self, batch):
        if self.hparams.regularization:
            out = self(**batch, output_attentions=True, return_dict=True)
            loss, logits, attentions = out["loss"], out["logits"], out["attentions"]

            if self.hparams.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                labels = batch["labels"]
                loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))

            info_vectors = attentions
            negative_entropy = compute_negative_entropy(
                info_vectors, batch["attention_mask"]
            )
            reg_loss = self.hparams.reg_strength * negative_entropy
            return loss, logits, negative_entropy, reg_loss

        else:
            out = self(**batch, return_dict=True)
            loss, logits = out["loss"], out["logits"]
            if self.hparams.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                labels = batch["labels"]
                loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
            return loss, logits

    def training_step(self, batch, batch_idx):
        if self.hparams.regularization:
            loss, logits, negative_entropy, reg_loss = self.forward_pass(batch)
            self.log("train_class_loss", loss, prog_bar=True)
            self.log("train_reg_loss", reg_loss, prog_bar=True)
            self.log("entropy", -negative_entropy)
            loss += reg_loss
        else:
            loss, logits = self.forward_pass(batch)
            self.log("train_class_loss", loss, prog_bar=True)

        y_true = batch["labels"]
        y_pred = logits.argmax(-1)

        self.train_acc(y_pred, y_true)
        self.train_F1(y_pred, y_true)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_F1", self.train_F1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.regularization:
            loss, logits, negative_entropy, reg_loss = self.forward_pass(batch)

            self.log("val_class_loss", loss, sync_dist=True)
            self.log("entropy", -negative_entropy, on_step=False, on_epoch=True)
            loss += reg_loss
        else:
            loss, logits = self.forward_pass(batch)
            self.log("val_class_loss", loss, sync_dist=True)

        y_true = batch["labels"]
        y_pred = logits.argmax(-1)

        self.val_acc(y_pred, y_true)
        self.val_F1(y_pred, y_true)

        # self.log("val_loss", loss, on_step=True, on_epoch=False)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val_F1", self.val_F1, on_step=False, on_epoch=True)

        if self.hparams.regularization:
            return {"val_loss": loss, "val_reg_loss": reg_loss}
        else:
            return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        btc_losses = torch.stack([x["val_loss"] for x in outputs])
        if self.hparams.regularization:
            reg_losses = torch.stack([x["val_reg_loss"] for x in outputs])

        if self.trainer.use_ddp:
            btc_losses = self.all_gather(btc_losses)

            if self.hparams.regularization:
                reg_losses = self.all_gather(reg_losses)

        self.log("val_loss", btc_losses.mean(), on_step=False, sync_dist=True)
        if self.hparams.regularization:
            self.log("val_reg_loss", reg_losses.mean(), on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        if self.hparams.regularization:
            loss, logits, negative_entropy, reg_loss = self.forward_pass(batch)
            loss += reg_loss
        else:
            loss, logits = self.forward_pass(batch)

        y_true = batch["labels"]
        y_pred = logits.argmax(-1)

        self.log("test_loss", loss, sync_dist=True)

        self.test_acc(y_pred, y_true)
        self.test_F1(y_pred, y_true)
        self.test_prec(y_pred, y_true)
        self.test_rec(y_pred, y_true)

        self.log("test_acc", self.test_acc)
        self.log("test_F1", self.test_F1)
        self.log("test_prec", self.test_prec)
        self.log("test_rec", self.test_rec)

    def configure_optimizers(self):
        # This code is taken from:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102

        # Don't apply weight decay to any parameters whose names include these tokens.
        # (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
        no_decay = ["bias", "LayerNorm.weight"]

        # Separate the `weight` parameters from the `bias` parameters.
        # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
        # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
        grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": self.hparams.weight_decay,
            },
            # Filter for parameters which *do* include those.
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        optimizer = AdamW(grouped_parameters, lr=self.hparams.learning_rate)

        if self.hparams.warmup_train_perc and self.hparams.train_steps_count:
            ws = int(self.hparams.warmup_train_perc * self.hparams.train_steps_count)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=ws,
                num_training_steps=self.hparams.train_steps_count,
            )

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def get_backbone(self):
        return self.model

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


def compute_negative_entropy(
    inputs: tuple, attention_mask: torch.Tensor, return_values=False
):
    """Compute the negative entropy across layers of a network for given inputs.

    Args:
        - input: tuple. Tuple of length num_layers. Each item should be in the form: BHSS
        - attention_mask. Tensor with dim: BS
    """
    inputs = torch.stack(inputs)  #  LayersBatchHeadsSeqlenSeqlen
    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form LBHSS"

    #  average over attention heads
    pool_heads = inputs.mean(2)

    batch_size = pool_heads.shape[1]
    samples_entropy = list()
    neg_entropies = list()
    for b in range(batch_size):
        #  get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = pool_heads[:, b, mask.bool(), :]
        sample = sample[:, :, mask.bool()]

        #  get the negative entropy for each non-padded token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)
        if return_values:
            neg_entropies.append(neg_entropy.detach())

        #  get the "average entropy" that traverses the layer
        mean_entropy = neg_entropy.mean(-1)

        #  store the sum across all the layers
        samples_entropy.append(mean_entropy.sum(0))

    # average over the batch
    final_entropy = torch.stack(samples_entropy).mean()
    if return_values:
        return final_entropy, neg_entropies
    else:
        return final_entropy


SUPPORTED_MODELS = [
    "bert-base-uncased",
    "bert-base-multilingual-uncased",
    "dbmdz/bert-base-italian-uncased",
]


@click.command()
@click.option("--src_model", type=str, required=True)
@click.option("--output_dir", type=str, default="./dumps")
@click.option("--training_dataset", type=str, default="wiki")
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=0)
@click.option("--seed", type=int, default=42)
@click.option("--max_epochs", type=int, default=20)
@click.option("--gpus", type=int, default=0)
@click.option("--accelerator", type=str, default=None)
@click.option("--max_seq_length", type=int, default=None)
@click.option("--learning_rate", type=float, default=2e-5)
@click.option("--early_stop_epochs", type=int, default=5)
@click.option("--regularization", type=str, default=None)
@click.option("--reg_strength", type=float, default=0.01)
@click.option("--weight_decay", type=float, default=0.0)
@click.option("--warmup_train_perc", type=float, default=None, help="Value [0,1]")
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--precision", type=int, default=32)
@click.option("--run_test", is_flag=True)
@click.option("--pin_memory", is_flag=True)
@click.option("--log_every_n_steps", type=int, default=50)
@click.option("--monitor", type=str, default="val_loss")
@click.option("--checkpoint_every_n_epochs", type=int, default=None)
@click.option("--save_transformers_model", is_flag=True)
@click.option("--ckpt_save_top_k", type=int, default=1)
@click.option("--resume_from_checkpoint", type=str, default=None)
@click.option("--balanced_loss", is_flag=True)
def main(
    src_model,
    output_dir,
    training_dataset,
    batch_size,
    num_workers,
    seed,
    max_epochs,
    gpus,
    accelerator,
    max_seq_length,
    learning_rate,
    early_stop_epochs,
    regularization,
    reg_strength,
    weight_decay,
    warmup_train_perc,
    accumulate_grad_batches,
    precision,
    run_test,
    pin_memory,
    log_every_n_steps,
    monitor,
    checkpoint_every_n_epochs,
    save_transformers_model,
    ckpt_save_top_k,
    resume_from_checkpoint,
    balanced_loss
):
    hparams = locals()
    pl.seed_everything(seed)

    model_name = None
    if src_model in SUPPORTED_MODELS:
        if not regularization:
            model_name = f"vanillabert-{training_dataset}-{seed}"
            experiment_name = f"vanillabert-{training_dataset}"
        elif regularization == "entropy":
            model_name = f"entropybert-{training_dataset}-{seed}-{reg_strength}"
            experiment_name = f"entropybert-{training_dataset}"
        elif regularization == "norm":
            model_name = f"normbert-{training_dataset}-{seed}-{reg_strength}"
            experiment_name = f"normbert-{training_dataset}"
    else:
        raise ValueError(f"src_model is not supported {src_model}")

    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, model_name)

    # logic to resume from checkpoint
    if os.path.exists(model_dir):
        if not resume_from_checkpoint:
            logger.info(
                f"The model {model_name} already exists and training was completed. Skipping..."
            )
            return
        else:
            ckpt_path = os.path.join(model_dir, resume_from_checkpoint)
            if os.path.exists(ckpt_path):
                logger.info(
                    f"The model {model_name} already exists but training was not completed. Resuming from {resume_from_checkpoint}..."
                )
                resume_from_checkpoint = ckpt_path
            else:
                logging.error(f"{ckpt_path} doesn't exist. Aborting.")
                return

    tokenizer = AutoTokenizer.from_pretrained(src_model)


    # logging.info("Tokenizing sets...")
    # tok_train = TokenizedDataset(train, tokenizer, max_seq_length, load_tokenized=True)
    # tok_val = TokenizedDataset(val, tokenizer, max_seq_length, load_tokenized=True)
    # tok_test = TokenizedDataset(test, tokenizer, max_seq_length, load_tokenized=True)
    # logging.info("Tokenization completed")

    # logging.info(f"TRAIN: {len(tok_train)}")
    # logging.info(f"VAL: {len(tok_val)}")
    # logging.info(f"TEST: {len(tok_test)}")

    # train_loader = DataLoader(
    #     tok_train,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     shuffle=True,
    # )
    # val_loader = DataLoader(
    #     tok_val, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    # )
    # test_loader = DataLoader(
    #     tok_test, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    # )

    dataset_module = TokenizerDataModule(
        dataset_name=training_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
        load_pre_tokenized=True,
        store_pre_tokenized=True,
    )

    # check if linear lr warmup is required
    train_steps_count = None
    if warmup_train_perc:
        logger.info(f"Warmup linear LR requested with {warmup_train_perc}")
        train_steps_count = (
            int(dataset_module.train_steps / accumulate_grad_batches) * max_epochs
        )
        logger.info(f"Total training steps: {train_steps_count}")
        if gpus and gpus > 0:
            train_steps_count = train_steps_count // gpus
        logger.info(f"Total training steps (gpu-normalized): {train_steps_count}")

    if balanced_loss:
        train, val, test = get_dataset_by_name(training_dataset)
        labels_count = pd.Series(train.labels).value_counts()
        labels_count = labels_count / len(train.labels)
        labels_count = 1 - labels_count
        labels_count = labels_count.sort_index()
        class_weights = torch.Tensor(labels_count)
        logger.info(f"Class weights: {class_weights}")
    else:
        class_weights = None
        
    #  Instantiate a LM and create the experiment accordingly
    model = LMForSequenceClassification(
        src_model,
        learning_rate,
        regularization,
        reg_strength,
        weight_decay=weight_decay,
        warmup_train_perc=warmup_train_perc,
        train_steps_count=train_steps_count,
        class_weights=class_weights
    )

    # set some training stuff (loggers, callback)
    loggers = list()
    if "COMET_API_KEY" in os.environ:
        comet_logger = pl.loggers.CometLogger(
            api_key=os.environ["COMET_API_KEY"],
            project_name="unbias-text-classifiers",  # Optional
            experiment_name=experiment_name,  # Optional
            log_code=False,
            log_graph=False,
        )
        comet_logger.experiment.add_tag("training")
        comet_logger.log_hyperparams(hparams)
        loggers.append(comet_logger)

    #  define training callbacks
    callbacks = list()
    if early_stop_epochs > 0:
        early_stopping = pl.callbacks.EarlyStopping(monitor, patience=early_stop_epochs)
        callbacks.append(early_stopping)

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        dirpath=model_dir,
        save_last=True,
        save_top_k=ckpt_save_top_k,
        filename="PL-{epoch}-{val_loss:.3f}-{train_loss:.3f}",
    )

    if checkpoint_every_n_epochs:
        from custom_callbacks import CheckpointEveryNEpochs

        ckpt_n_epochs = CheckpointEveryNEpochs(checkpoint_every_n_epochs)
        callbacks.append(ckpt_n_epochs)

    lr_monitor = pl.callbacks.LearningRateMonitor()
    callbacks.append(model_checkpoint)
    callbacks.append(lr_monitor)

    trainer = pl.Trainer(
        gpus=gpus,
        accelerator=accelerator,
        max_epochs=max_epochs,
        logger=loggers,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        resume_from_checkpoint=resume_from_checkpoint,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=1
        # plugins=pl.plugins.DDPPlugin(find_unused_parameters=True),
    )

    trainer.fit(model, datamodule=dataset_module)

    logging.info(f"Best model path: {model_checkpoint.best_model_path}")
    logging.info(f"Best model val_loss: {model_checkpoint.best_model_score}")

    #  print(trainer.logger[0].experiment.get_key())
    if run_test:
        if "COMET_API_KEY" in os.environ:
            trainer.logger = None
        # test on the dataset in-distribution
        trainer.test(datamodule=dataset_module, ckpt_path="best")

    if save_transformers_model:
        #  Save the tokenizer and the backbone LM with HuggingFace's serialization.
        #  To avoid mixing PL's and HuggingFace's serialization:
        #  https://github.com/PyTorchLightning/pytorch-lightning/issues/3096#issuecomment-686877242
        best_PL = LMForSequenceClassification.load_from_checkpoint(
            model_checkpoint.best_model_path
        )
        best_PL.get_backbone().save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    #  TODO resume_from_checkpoint logic
    #  logger.info("Simulation completed. Removing last.ckpt...")
    #  if early_stop_epochs > 0:
    #      if os.path.exists(os.path.join(model_dir, "last.ckpt")):
    #          os.remove(os.path.join(model_dir, "last.ckpt"))
    #          logger.info("Last checkpoint removed.")


if __name__ == "__main__":
    main()

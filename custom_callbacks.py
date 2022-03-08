import os
import pytorch_lightning as pl
import logging

logging.basicConfig(format="%(asctime)s:%(module)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointEveryNEpochs(pl.Callback):
    """
    Save a checkpoint every N epochs, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(self, save_epoch_frequency: int, skip_first: bool = True):
        self.save_epoch_frequency = save_epoch_frequency
        self.skip_first = skip_first

    def on_validation_epoch_end(self, trainer, pl_module):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if epoch == 0 and self.skip_first:
            return

        # global_step = trainer.global_step
        if (epoch + 1) % self.save_epoch_frequency == 0:
            logger.info(f"Dumping checkpoint at epoch {epoch}")
            metrics = trainer.logged_metrics
            filename = f"PL-epoch={epoch}-val_loss={metrics['val_loss']:.3f}-val_reg_loss={metrics['val_reg_loss']:.3f}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

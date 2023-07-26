import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import AnimalsDataset
from engine import LightningInception, dummy_forward


def main():
    torch.set_float32_matmul_precision('high')

    # seed
    pl.seed_everything(1, True)

    # dataset initialization
    dataset_test = AnimalsDataset('data/10animals_split/test')
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=64,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True
    )

    # model initialization 
    model = LightningInception.load_from_checkpoint('model_log/test/lightning_logs/version_1/checkpoints/epoch=9-step=360.ckpt')

    # training 
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        strategy='ddp',
        default_root_dir=f'model_log/test',
        enable_checkpointing=True,
        logger=True,
        max_epochs=10,
        check_val_every_n_epoch=1,
        log_every_n_steps=0
        # val_check_interval=0
    )

    trainer.test(model, dataloader_test)
    


if __name__ == '__main__':
    main()
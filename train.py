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
    dataset_train = AnimalsDataset('data/10animals_split/train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True
    )

    dataset_valid = AnimalsDataset('data/10animals_split/valid')
    dataloader_valid = DataLoader(
        dataset_valid,
        batch_size=64,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True
    )

    # model initialization 
    model = LightningInception(10, 64)
    dummy_forward(model)


    # training 
    trainer = pl.Trainer(
        accelerator='gpu',
        devices='auto',
        strategy='ddp',
        default_root_dir=f'model_log/test',
        enable_checkpointing=True,
        logger=True,
        max_epochs=10,
        check_val_every_n_epoch=1,
        log_every_n_steps=0
        # val_check_interval=0
    )

    trainer.fit(model, dataloader_train, dataloader_valid)


    resume_training = False
    if resume_training:
        trainer.fit(model, ckpt_path="somecheckpoint.ckpt")
    


if __name__ == '__main__':
    main()
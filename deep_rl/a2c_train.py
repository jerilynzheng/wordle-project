"""Advantage Actor Critic (A2C)"""
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from a2c.module import AdvantageActorCritic


def cli_main() -> None:
    parser = ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = AdvantageActorCritic.add_model_specific_args(parser)

    parser.set_defaults(max_epochs=1000, every_n_epochs=1)
    parser.add_argument("--load_model", type=str, help="load model from path")
    parser.add_argument("--save_model", type=str, help="save model to path")

    args = parser.parse_args()

    with wandb.init(project='wordle-solver'):
        wandb.config.update(args)

        model = AdvantageActorCritic(**args.__dict__)
        if args.load_model is not None:
            model.load_state_dict(torch.load(args.load_model))
        # save checkpoints based on avg_reward
        checkpoint_callback = ModelCheckpoint(every_n_epochs=args.every_n_epochs)

        seed_everything(123)

        trainer = Trainer.from_argparse_args(args, limit_train_batches=model.batches_per_epoch,
                                             deterministic=True, callbacks=checkpoint_callback)
        trainer.fit(model)
        if args.save_model is not None:
            torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    cli_main()

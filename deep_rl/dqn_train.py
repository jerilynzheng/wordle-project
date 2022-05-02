"""Deep Q Network (DQN)"""
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from dqn.module import DQNLightning, SaveBufferCallback


def cli_main() -> None:
    parser = ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=10000,
                        checkpoint_every_n_epochs=1000)

    # model args
    parser = DQNLightning.add_model_specific_args(parser)

    # program args

    args = parser.parse_args()

    with wandb.init(project='wordle-solver'):
        wandb.config.update(args)

        model = DQNLightning(**args.__dict__)
        # save checkpoints based on avg_reward
        checkpoint_callback = ModelCheckpoint(every_n_train_steps=args.checkpoint_every_n_epochs)

        seed_everything(123)

        save_buffer_callback = SaveBufferCallback(buffer=model.dataset.winners, filename='sequence_buffer.pkl')

        trainer = Trainer.from_argparse_args(args, deterministic=True,
                                             callbacks=[checkpoint_callback, save_buffer_callback])
        trainer.fit(model)


if __name__ == '__main__':
    cli_main()

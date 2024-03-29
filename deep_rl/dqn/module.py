from dataclasses import dataclass
from typing import Tuple, List, Any

from collections import OrderedDict
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

import dqn
from dqn.agent import Agent
from dqn.experience import SequenceReplay, RLDataset, Experience

import wordle.state

import gym


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
            self,
            initialize_winning_replays: str = None,
            deep_q_network: str = 'SumChars',
            batch_size: int = 1024,
            lr: float = 1e-2,
            weight_decay: float = 1.e-4,
            env: str = "WordleEnv100-v0",
            gamma: float = 0.9,
            sync_rate: int = 10,
            replay_size: int = 1000,
            hidden_size: int = 256,
            num_workers: int = 0,
            warm_start_size: int = 1000,
            warm_start_steps: int = 1000,
            eps_last_frame: int = 10000,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            episode_length: int = 512,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()     # saves all in the __init__() argument list

        self.writer = SummaryWriter()
        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self._winning_steps = 0
        self._wins = 0
        self._losses = 0
        self._rewards = 0

        print("dqn:", self.env.spec.id, self.env.spec.max_episode_steps, n_actions, obs_size)

        self.net = dqn.construct(
            self.hparams.deep_q_network, obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size, word_list=self.env.words)
        self.target_net = dqn.construct(
            self.hparams.deep_q_network, obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size, word_list=self.env.words)

        self.dataset = RLDataset(
            winners=SequenceReplay(self.hparams.replay_size//2, self.hparams.initialize_winning_replays),
            losers=SequenceReplay(self.hparams.replay_size//2),
            sample_size=self.hparams.episode_length)

        self.agent = Agent(self.net, self.env.action_space)
        self.state = self.env.reset()
        self.total_reward = 0
        self.episode_reward = 0
        self.total_games_played = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.play_game(epsilon=1., device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    @torch.no_grad()
    def play_game(
            self,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Tuple[float, bool]:
        done = False
        cur_seq = list()
        reward = 0
        while not done:
            exp = self.play_step(epsilon, device)
            done = exp.done
            reward = exp.reward
            cur_seq.append(exp)

        winning_steps = self.env.max_turns - wordle.state.remaining_steps(self.state)
        if reward > 0:
            self.dataset.winners.append(cur_seq)
        else:
            self.dataset.losers.append(cur_seq)
        self.state = self.env.reset()

        return reward, winning_steps

    def play_step(
            self,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Experience:
        action = self.agent.get_action(self.state, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state.copy(), action, reward, done, new_state.copy(), self.env.goal_word)

        self.state = new_state
        return exp

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step / self.hparams.eps_last_frame,
            )

        # step through environment with agent
        with torch.no_grad():
            reward, winning_steps = self.play_game(epsilon, device)
        self.total_games_played += 1
        if reward > 0:
            self._wins += 1
            self._winning_steps += winning_steps
        else:
            self._losses += 1

        self._rewards += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss.detach(),
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        if self.global_step % 100 == 0:
            if len(self.dataset.winners) > 0:
                winner = self.dataset.winners.buffer[-1]
                game = f"goal: {self.env.words[winner[0].goal_id]}\n"
                for i, xp in enumerate(winner):
                    game += f"{i}: {self.env.words[xp.action]}\n"
                self.writer.add_text("game sample/winner", game, global_step=self.global_step)
            if len(self.dataset.losers) > 0:
                loser = self.dataset.losers.buffer[-1]
                game = f"goal: {self.env.words[loser[0].goal_id]}\n"
                for i, xp in enumerate(loser):
                    game += f"{i}: {self.env.words[xp.action]}\n"
                self.writer.add_text("game sample/loser", game, global_step=self.global_step)
            self.writer.add_scalar("train_loss", loss, global_step=self.global_step)
            self.writer.add_scalar("total_games_played", self.total_games_played, global_step=self.global_step)

            self.writer.add_scalar("winner_buffer", len(self.dataset.winners), global_step=self.global_step)
            self.writer.add_scalar("loser_buffer", len(self.dataset.losers), global_step=self.global_step)

            self.writer.add_scalar("lose_ratio", self._losses/(self._wins+self._losses), global_step=self.global_step)
            self.writer.add_scalar("wins", self._wins, global_step=self.global_step)
            self.writer.add_scalar("reward_per_game", self._rewards / (self._wins+self._losses), global_step=self.global_step)
            if self._wins > 0:
                self.writer.add_scalar("avg_winning_turns", self._winning_steps/self._wins, global_step=self.global_step)
            self._winning_steps = 0
            self._wins = 0
            self._losses = 0
            self._rewards = 0

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(arg_parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments for DQN model.
        Args:
            arg_parser: the current argument parser to add to
        Returns:
            arg_parser with model specific cargs added
        """

        arg_parser.add_argument("--initialize_winning_replays", type=str, default=None, help="initialize winning replays")
        arg_parser.add_argument("--env", type=str, default="WordleEnv100-v0", help="gym environment tag")
        arg_parser.add_argument("--deep_q_network", type=str, default="SumChars", help="Network to use")
        arg_parser.add_argument("--checkpoint_every_n_epochs", type=int, default=10000, help="checkpoint every n epochs")
        arg_parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
        arg_parser.add_argument("--num_workers", type=int, default=0, help="number of workers")
        arg_parser.add_argument("--replay_size", type=int, default=1000, help="Size of replay buffer(s)")
        arg_parser.add_argument("--hidden_size", type=int, default=256, help="Width of hidden layers")
        arg_parser.add_argument("--sync_rate", type=int, default=100, help="how many frames do we update the target network")
        arg_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        arg_parser.add_argument("--weight_decay", type=float, default=1e-5, help="Optimizer weight decay regularization.")
        arg_parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
        arg_parser.add_argument("--warm_start_size", type=int, default=1000, help="how many samples do we use to fill our buffer at the start of training")
        arg_parser.add_argument("--warm_start_steps", type=int, default=1000, help="max episode reward in the environment")
        arg_parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
        arg_parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
        arg_parser.add_argument("--eps_last_frame", type=int, default=10000, help="what frame should epsilon stop decaying")
        arg_parser.add_argument("--episode_length", type=int, default=512, help="max length of an episode")

        return arg_parser


@dataclass
class SaveBufferCallback(Callback):
    buffer: SequenceReplay
    filename: str

    def on_train_end(self, trainer, pl_module):
        path = f'{trainer.log_dir}/checkpoints'
        fname = self.filename,
        self.buffer.save(f'{path}/{fname}')


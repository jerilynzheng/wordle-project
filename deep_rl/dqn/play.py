from typing import Tuple, List

import wordle.state
from wordle.wordle import WordleEnvBase
from .module import DQNLightning
from .agent import Agent


def load_from_checkpoint(
        checkpoint: str
) -> Tuple[DQNLightning, Agent, WordleEnvBase]:
    """
    :param checkpoint:
    :return:
    """
    model = DQNLightning.load_from_checkpoint(checkpoint)
    env = model.env
    agent = Agent(model.net, action_space=env.action_space)

    return model, agent, env


def suggest(
        agent,
        env,
        sequence: List[Tuple[str, List[int]]],
) -> str:
    """
    Given a list of words and masks, return the next suggested word

    :param agent:
    :param env:
    :param sequence: History of moves and outcomes until now
    :return:
    """
    state = env.reset()
    for word, mask in sequence:
        word = word.upper()
        assert word in env.words, f'{word} not in allowed words!'
        assert all(i in (0, 1, 2) for i in mask)
        assert len(mask) == 5

        state = wordle.state.update_from_mask(state, word, mask)

    return env.words[agent.get_action(state, 0.0, "cpu")]


def goal(
        agent,
        env,
        goal_word: str,
) -> Tuple[bool, List[Tuple[str, int]]]:
    state = env.reset()
    try:
        env.set_goal_word(goal_word.upper())
    except:
        raise ValueError("Goal word", goal_word, "not found in env words!")

    outcomes = []
    win = False
    for i in range(env.max_turns):
        action = agent.get_action(state, 0.0, "cpu")
        state, reward, done, _ = env.step(action)
        outcomes.append((env.words[action], reward))
        if done:
            if reward >= 0:
                win = True
            break

    return win, outcomes

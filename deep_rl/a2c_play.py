import fire
import torch

import a2c.play


def main(
        checkpoint: str,
        mode: str = 'goal',
        env: str = 'WordleEnv100-v0',
        model_path: str = None
):
    print("Loading from checkpoint", checkpoint, "...")
    model, agent, env = a2c.play.load_from_checkpoint(checkpoint, env)
    print("Got env with", len(env.words), "words!")

    if mode == 'goal':
        goal(agent, env)
    elif mode == 'debug':
        goal(agent, env, debug=True)
    elif mode == 'suggest':
        suggest(agent, env)
    elif mode == 'evaluate':
        evaluate(agent, env)
    elif mode == 'export' and model_path is not None:
        save(model, model_path)
    else:
        print("valid modes are goal, suggest, evaluate")


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def suggest(agent, env):
    print("Interactive mode")
    print("When I ask for <word> <mask>, give me the word you entered\n"
          "and the result, example: stare 21021\n"
          "  where 0 = not in word, 1 = somewhere in this word, 2 = in this spot")
    while True:
        print("Alright, a new game!")
        word_masks = []
        while True:
            guess = a2c.play.suggest(agent, env, word_masks)
            print(f"I suggest", guess)
            word_mask = input("<mask>, <word mask>, or done: ")
            if word_mask.lower() == 'done':
                break
            try:
                word_mask = word_mask.strip().split(' ')
                if len(word_mask) == 2:
                    word, mask = word_mask
                else:
                    word = guess
                    mask = word_mask[0]
                word = word.upper()
                assert word in env.words
                mask_arr = [int(i) for i in mask]
                assert all(i in (0, 1, 2) for i in mask_arr)
                assert len(mask_arr) == 5

                word_masks.append((word, mask_arr))
            except:
                print(f"Failed to parse {word_mask}!")
                continue


def state_string(state):
    result = f'remaining turns: {state[0]}\n'
    tried = state[1:27]
    x = state[27:].reshape(26, 5, 3).argmax(axis=2)
    return result + '\n'.join(
        f'{chr(ord("A") + i)}: {tried[i]}: {"".join(str(y) for y in row)}' for i, row in enumerate(x)
    )


def goal(agent, env, debug=False):
    print("Goal word mode")
    while True:
        goal_word = input("Give me a goal word: ")
        try:
            win, outcomes = a2c.play.goal(agent, env, goal_word)

            i = 0
            for guess, reward, state in outcomes:
                if debug:
                    print(f"Turn {i+1}: {guess}, reward ({reward})")
                    print(state_string(state))
                i += 1

            if win:
                print(f"Done! took {i} guesses!")
            else:
                print(f"LOSE! took {i} guesses!")
        except Exception as e:
            print(e)
            continue


def evaluate(agent, env):
    print("Evaluation mode")
    n_wins = 0
    n_guesses = 0
    n_win_guesses = 0
    N = len(env.words)
    for goal_word in env.words[:N]:
        win, outcomes = a2c.play.goal(agent, env, goal_word)
        if win:
            n_wins += 1
            n_win_guesses += len(outcomes)
        else:
            print("Lost!", goal_word, outcomes)
        n_guesses += len(outcomes)

    print(f"Evaluation complete, won {n_wins/N*100}% and took {n_win_guesses/n_wins} guesses per win, "
          f"{n_guesses / N} including losses.")


if __name__ == '__main__':
    fire.Fire(main)
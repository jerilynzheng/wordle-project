import fire

from dqn import play


def main(
        checkpoint: str,
        mode: str = 'goal',
):
    print("Loading from checkpoint", checkpoint, "...")
    dqn, agent, env = play.load_from_checkpoint(checkpoint)
    print("Got env with", len(env.words), "words!")

    if mode == 'goal':
        goal(dqn, agent, env)
    elif mode == 'suggest':
        suggest(dqn, agent, env)
    elif mode == 'evaluate':
        evaluate(dqn, agent, env)


def suggest(dqn, agent, env):
    print("Interactive mode")
    print("When I ask for <word> <mask>, give me the word you entered\n"
          "and the result, example: stare 21021\n"
          "  where 0 = not in word, 1 = somewhere in this word, 2 = in this spot")
    while True:
        print("Alright, a new game!")
        word_masks = []
        while True:
            guess = dqn.suggest(word_masks)
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
                assert word in dqn.env.words
                mask_arr = [int(i) for i in mask]
                assert all(i in (0, 1, 2) for i in mask_arr)
                assert len(mask_arr) == 5

                word_masks.append((word, mask_arr))
            except:
                print(f"Failed to parse {word_mask}!")
                continue


def goal(dqn, agent, env):
    print("Goal word mode")
    while True:
        goal_word = input("Give me a goal word: ")
        try:
            win, outcomes = play.goal(agent, env, goal_word)

            i = 0
            for guess, reward in outcomes:
                print(f"Turn {i+1}: {guess}, reward ({reward})")
                i += 1

            if win:
                print(f"Done! took {i} guesses!")
            else:
                print(f"LOSE! took {i} guesses!")
        except Exception as e:
            print(e)
            continue


def evaluate(dqn, agent, env):
    print("Evaluation mode")
    n_wins = 0
    n_guesses = 0
    n_win_guesses = 0
    N = len(env.words)
    for goal_word in env.words[:N]:
        win, outcomes = play.goal(agent, env, goal_word)
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

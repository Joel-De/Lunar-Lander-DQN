import numpy.random
import pandas
import torch
import argparse
import os
import gym
import logging
import pandas as pd

from validate import validate
from AgentModule import AgentModule

_CURRDIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


def train(enviornment, agent, episodes, valInterval, val, earlyStop):
    metrics = {"Train Episode": [], "Train Score": [], "50-Episode Average": []}
    displayTraining = False  # Option to show the environment while training, - greatly slows down training
    eps = 1.0
    epsEnd = 0.02
    epsDecay = 0.995
    stepCount = 1000

    for episode in range(1, episodes + 1):
        score = 0
        currentState = enviornment.reset()
        for step in range(stepCount):
            action = agent.getNextAction(currentState, eps)
            nextState, reward, end, _ = enviornment.step(action)

            if displayTraining:
                enviornment.render()

            agent.storeData(currentState, action, reward, nextState, end)
            currentState = nextState
            score += reward

            if end:
                break

        metrics["Train Episode"].append(episode)
        metrics["Train Score"].append(score)

        # Computes average for last 50 runs
        if episode < 50:
            metrics["50-Episode Average"].append(sum(metrics["Train Score"]) / len(metrics["Train Score"]))
        else:
            metrics["50-Episode Average"].append(sum(metrics["Train Score"][-50:]) / 50)
        print(f"\rEpisode: {episode}\t Score: {score}\t50-Episode Average:{metrics['50-Episode Average'][-1]}", end='')
        if val and (episode % valInterval) == 0:
            print('\n')
            res = validate(enviornment, agent, 10)
            logging.info(f"Validation Completed, average score of {sum(res) / 10} over {10} episodes")

        if (earlyStop is not None) and (metrics["50-Episode Average"][-1] >= earlyStop):
            break
        eps = max(epsEnd, eps * epsDecay)  # Update epsilon
    if displayTraining:
        enviornment.close()

    return pandas.DataFrame.from_dict(metrics)


def main():
    parser = argparse.ArgumentParser(description="Training Script for Lunar Lander Simulation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate to be used")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes to train for")
    parser.add_argument("--val", type=bool, default=False, help="Whether to validate or not during training")
    parser.add_argument("--val_interval", type=int, default=100, help="Interval at which to perform validation")
    parser.add_argument("--save_metrics", type=bool, default=True, help="Whether to save training metrics")
    parser.add_argument("--save_models", type=bool, default=True, help="Whether to save models")
    parser.add_argument("--cpu", type=bool, default=False,
                        help="Uses CPU regardless of whether CUDA capable device is present")
    parser.add_argument("--checkpoint", type=str, default=None, help="Location of pretrained saved checkpoints")
    parser.add_argument("--early_stop", type=int, default=200, help="Threshold for 50 episode average after which training will stop ")
    args = parser.parse_args()
    saveDIR = "SavedModels"
    if not os.path.exists(os.path.join(_CURRDIR, saveDIR)):
        logging.info(f"Created SavedModels at {_CURRDIR}")
        os.mkdir(os.path.join(_CURRDIR, saveDIR))
    else:
        logging.info(f"Model Directory exists")

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make("LunarLander-v2")
    logging.info("Created Lunar Lander Environment")

    agent = AgentModule(args.batch_size, args.learning_rate, 0.99, 2000, device)

    if args.checkpoint is not None:
        loadedmodel = torch.load(args.checkpoint)
        agent.loadWeights(loadedmodel)
        logging.info(f"Loaded Model Weights from {args.checkpoint}")

    trainMetrics = train(env, agent, args.episodes, args.val_interval, args.val, args.early_stop)

    if args.save_metrics:
        trainMetrics.to_csv(os.path.join(_CURRDIR, saveDIR, "Metrics.csv"), index=False)
    model = agent.getWeights()
    if args.save_models:
        torch.save(model.state_dict(), os.path.join(_CURRDIR, saveDIR, "LunarLanderModel.pt"))


if __name__ == '__main__':
    print(numpy.random.get_state()[1][0])
    print(torch.seed())
    main()

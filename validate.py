import logging
import torch
import argparse
import gym
import os
import time

from AgentModule import AgentModule

logging.basicConfig(level=logging.INFO)


def validate(enviornment, agent, sessionCount=10):
    scoreList = []
    for i in range(sessionCount):
        state = enviornment.reset()
        rewardTotal = 0
        for idx_step in range(2000):
            action = agent.getNextAction(state, epsilon=0)
            enviornment.render()
            state, reward, end, _ = enviornment.step(action)
            rewardTotal += reward
            if end:
                break
        logging.info(f"Validation episode {i + 1} Score:{rewardTotal}")
        scoreList.append(rewardTotal)
    enviornment.close()
    time.sleep(1)
    return scoreList


def main():
    parser = argparse.ArgumentParser(description="Validation/Testing Script for Lunar Lander Simulation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to train for")
    parser.add_argument("--cpu", type=bool, default=False, help="Uses CPU regardless of whether CUDA capable device is present")
    parser.add_argument("--model_weights", type=str, default="SavedModels/LunarLanderModel.pt", help="Directory to Model weights")

    args = parser.parse_args()

    if not os.path.exists(args.model_weights):
        raise FileNotFoundError(args.model_weights)

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make("LunarLander-v2")
    logging.info("Created Lunar Lander Environment")

    agent = AgentModule(None, None, None, None, device, ValidationMode=True)
    savedModel = torch.load(args.model_weights)
    logging.info(f"Loaded Model Weights from {args.model_weights}")
    agent.loadWeights(savedModel)
    scores = validate(env, agent, sessionCount=args.episodes)
    logging.info(f"Validation Completed, average score of {sum(scores) / args.episodes} over {args.episodes} episodes")


if __name__ == '__main__':
    main()

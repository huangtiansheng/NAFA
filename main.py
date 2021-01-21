import numpy as np
import torch
from util.data_util import read_energy_data
from schedule.NAFA import NAFA_Agent
from schedule.best_fit import best_fit
from comp_offload import CompOffloadingEnv
from schedule.linUCB import LinUCBAgent
from schedule.worst_fit import worst_fit
from util.options import args_parser
import pickle


def save_data(env, args):
    output = open(
        'data/n_reject_low_power_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)), 'wb')
    pickle.dump(env.n_reject_low_power, output)
    output = open(
        'data/n_reject_conservation_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
        'wb')
    pickle.dump(env.n_reject_conservation, output)
    output = open(
        'data/n_reject_high_latency_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
        'wb')
    pickle.dump(env.n_reject_high_latency, output)
    output = open('data/total_latency_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
                  'wb')
    pickle.dump(env.total_latency, output)
    output = open('data/n_total_request_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
                  'wb')
    pickle.dump(env.n_total_request, output)
    output = open('data/day_rewards_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
                  'wb')
    pickle.dump(env.day_rewards, output)


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cpu")
    env = CompOffloadingEnv(args)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n
    immediate_reward = 0
    print(args.method)
    if args.method == "NAFA":
        agent = NAFA_Agent(args, env)
        # agent.load_weights("data/model_{}_{}.pkl".format(args.lambda_r,args.tradeoff))
        agent.train()
        # print("data/model_{}_{}_{}.pkl".format(args.lambda_r,args.tradeoff,str(args.trial)))
        agent.load_weights("data/model_{}_{}_{}.pkl".format(args.lambda_r, args.tradeoff, str(args.trial)))
    if args.method == "BF":  # best-fit
        agent = best_fit(env)
    if args.method == "WF":  # worst-fit
        agent = worst_fit(env)
    if args.method == "linUCB":
        agent = LinUCBAgent(args, env)
        agent.train()
        agent.load_model()
    episode = 0
    GHI_Data = read_energy_data(is_train=False)
    done = True
    accept = 0
    while episode < 1:
        if done:
            s = env.reset(is_train=False, simulation_start=0, simulation_end=300 * 24, GHI_Data=GHI_Data)
        action = agent.act(s)
        if action != 0:
            accept += 1
        s, r, done = env.step(action)
        if done:
            print("average accept ratio of {}".format(accept))
            print("average reward{}".format(np.mean(env.day_rewards)))
            episode += 1
    save_data(env, args)

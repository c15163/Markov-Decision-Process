import random as random
import numpy as np
import six
import sys
sys.modules['sklearn.externals.six'] = six
import hiive.mdptoolbox
from hiive.mdptoolbox import mdp, example
from hiive.mdptoolbox.openai import OpenAI_MDPToolbox
import gym
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.frozen_lake import generate_random_map
import re
import time
from matplotlib import pyplot as plt
from typing import List, Optional

plt.rc('font', size=14)  # 기본 폰트 크기
plt.rc('axes', labelsize=14)  # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=14)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=14)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=10)  # 범례 폰트 크기
plt.rc('figure', titlesize=18)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}
# codes are coming from https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py and https://github.com/hiive/hiivemdptoolbox/blob/master/hiive/mdptoolbox/openai.py
# class override to avoid "absorbing state"
class FrozenLakeEnvExtend(FrozenLakeEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self,render_mode: Optional[str] = None,desc=None,map_name="4x4",re=[1.0, -1.0, -0.1], is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        nA = 4
        nS = nrow * ncol
        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        def to_s(row, col):
            return row * ncol + col
        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            reward=re[2]
            #reward = float(newletter == b"G")
            if newletter == b"G":
                reward = re[0]
            elif newletter == b"H":
                reward = re[1]
            elif newletter in b"FS":
                reward = re[2]
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        self.render_mode = render_mode
        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

class OpenAI_MDPToolbox_modified(OpenAI_MDPToolbox):
    def __init__(self, env, render: bool = False, **kwargs):
        self.env=env
        self.env.reset()
        if render:
            self.env.render()
        self.transitions = self.env.P
        self.actions = int(re.findall(r'\d+', str(self.env.action_space))[0])
        self.states = int(re.findall(r'\d+', str(self.env.observation_space))[0])
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))
        #self.convert_PR()

    def convert_PR(self):
        for state in range(self.states):
            for action in range(self.actions):
                for i in range(len(self.transitions[state][action])):
                    tran_prob = self.transitions[state][action][i][0]
                    state_ = self.transitions[state][action][i][1]
                    self.R[state][action] += tran_prob * self.transitions[state][action][i][2]
                    self.P[action, state, state_] += tran_prob
        return self.P, self.R

#cite: https://gymnasium.farama.org/tutorials/blackjack_tutorial/#visualising-the-policy
def map_color():
    return {b'S': 'green', b'F': 'grey', b'H': 'red', b'G': 'blue'}

def map_color2():
    return 'grey'

def number_to_directions():
    return {3: '↑',2: '→',1: '↓',0: '←'}

def number_to_action():
    return {0: 'W',1: 'C'}

def plot_policy_map(title, policy, map_desc, map_color, number_to_directions):
    policy=np.array(policy).reshape(size, size)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'large'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(map_color[map_desc[i,j]])
            ax.add_patch(p)
            text = ax.text(x+0.5, y+0.5, number_to_directions[policy[i, j]], weight='bold', size=font_size, horizontalalignment='center', verticalalignment='center', color='w')

def plot_forest_policy_map(title, policy, s, map_color2, number_to_action):
    policy=np.array(policy).reshape(int(np.sqrt(s)), int(np.sqrt(s)))
    map_desc = np.zeros((int(np.sqrt(s)), int(np.sqrt(s))), dtype=int)
    if policy.shape[1] < 7:
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
        font_size = 20
        state_font_size = 12
    elif policy.shape[1] > 6 and policy.shape[1] < 11:
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
        font_size = 18
        state_font_size = 10
    else:
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
        font_size = 16
        state_font_size = 8
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(map_color2[map_desc[i,j]])
            ax.add_patch(p)
            text = ax.text(x+0.5, y+0.5, number_to_action[policy[i, j]], weight='bold', size=font_size, horizontalalignment='center', verticalalignment='center', color='w')
            text2 = ax.text(x+0.1, y+0.9, int(i*policy.shape[0]+j), size=state_font_size, horizontalalignment='left', verticalalignment='top', color='k')

# 1. small size Grid problem
np.random.seed(0)
size=5
random_map1 = generate_random_map(size=size)
map_env=FrozenLakeEnvExtend(desc=random_map1, re=[1.0, -1.0, -0.01], is_slippery=True)
desc=map_env.unwrapped.desc
P, R =OpenAI_MDPToolbox_modified(map_env).convert_PR()
iteration=10000
gammas=[0.7, 0.8, 0.9, 0.99, 0.999]
epsilons=[1e-5, 1e-7, 1e-10, 1e-12, 1e-15]

# 1-1. VI
gamma=0.999
a=np.zeros(len(epsilons))
b=np.zeros(len(epsilons))
for i, epsilon in enumerate(epsilons):
    solver = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=gamma, epsilon=epsilon, max_iter=iteration, skip_check=False, run_stat_frequency=True)
    result = solver.run()
    a[i] = result[-1]['Reward']
    b[i] = result[-1]['Time']
    print('VI-small-grid(epsilon): Reward when epsilon is {}: {}'.format(epsilon, result[-1]['Reward']))
    print('VI-small-grid(epsilon): Time when epsilon is {}: {}'.format(epsilon, result[-1]['Time']))
plt.figure(0)
plt.grid()
x=['1e-5', '1e-7', '1e-10', '1e-12', '1e-15']
plt.plot(x, a, marker='o')
plt.title('Reward of each epsilon')
plt.ylabel('Reward')
plt.xlabel('Epsilon Value')
plt.tight_layout()
plt.savefig('Fig0_reward_epsilon_VI.png')

plt.figure(0-1)
plt.grid()
x=['1e-5', '1e-7', '1e-10', '1e-12', '1e-15']
plt.plot(x, b, marker='o')
plt.title('Time of each epsilon')
plt.ylabel('Time(sec)')
plt.xlabel('Epsilon Value')
plt.tight_layout()
plt.savefig('Fig0-1_time_epsilon_VI.png')

epsilon=1e-10
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=False, run_stat_frequency=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Reward']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(1)
    plt.plot(a[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Reward of each gamma')
    plt.ylabel('Reward')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig1_reward_gamma_VI.png')
    plt.grid()
    plt.figure(2)
    plt.plot(b[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig2_mean_V_gamma_VI.png')
    plt.grid()
    plt.figure(3)
    plt.plot(c[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig3_time_gamma_VI.png')
    plt.grid()
    plt.figure(4)
    plt.plot(d[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.xlabel('Number of Iteration')
    plt.yscale('log', base=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig4_error_gamma_VI.png')
    print('VI-small-grid: Iteration number when gamma is {:.3f}: {}'.format(gamma, result[-1]['Iteration']))
    print('VI-small-grid: Reward when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Reward']))
    print('VI-small-grid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('VI-small-grid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of VI in small Grid problem'
solver=hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=0.999, epsilon=epsilon, max_iter=iteration, skip_check=False, run_stat_frequency=True)
solver.run()
policy=solver.policy
plot_policy_map(title, policy, desc, map_color(), number_to_directions())
plt.savefig('Fig4-1_map_visual_VI.png')
plt.close()

#1-2. PI
gamma=0.999
a=np.zeros(len(epsilons))
b=np.zeros(len(epsilons))
for i, epsilon in enumerate(epsilons):
    solver2=hiive.mdptoolbox.mdp.PolicyIterationModified(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=True)
    solver2.run()
    solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=solver2.iter, skip_check=True)
    #solver = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=iteration, skip_check=True)
    result = solver.run()
    a[i] = result[-1]['Reward']
    b[i] = result[-1]['Time']
    print('PI-small-grid(epsilon): Reward when epsilon is {}: {}'.format(epsilon, result[-1]['Reward']))
    print('PI-small-grid(epsilon): Time when epsilon is {}: {}'.format(epsilon, result[-1]['Time']))
plt.figure(0-2)
plt.grid()
x=['1e-5', '1e-7', '1e-10', '1e-12', '1e-15']
plt.plot(x, a, marker='o')
plt.title('Reward of each epsilon')
plt.ylabel('Reward')
plt.xlabel('Epsilon Value')
plt.tight_layout()
plt.savefig('Fig0-2_reward_epsilon_PI.png')
plt.close()

plt.figure(0-3)
plt.grid()
x=['1e-5', '1e-7', '1e-10', '1e-12', '1e-15']
plt.plot(x, b, marker='o')
plt.title('Time of each epsilon')
plt.ylabel('Time(sec)')
plt.xlabel('Epsilon Value')
plt.tight_layout()
plt.savefig('Fig0-3_time_epsilon_PI.png')
plt.close()

a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver2=hiive.mdptoolbox.mdp.PolicyIterationModified(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=True)
    solver2.run()
    solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=solver2.iter, skip_check=True)
    #solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=iteration, skip_check=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Reward']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(5)
    plt.plot(a[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Reward of each gamma')
    plt.ylabel('Reward')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig5_reward_gamma_PI.png')
    plt.grid()
    plt.figure(6)
    plt.plot(b[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig6_mean_V_gamma_PI.png')
    plt.grid()
    plt.figure(7)
    plt.plot(c[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig7_time_gamma_PI.png')
    plt.grid()
    plt.figure(8)
    plt.plot(d[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.xlabel('Number of Iteration')
    plt.yscale('log', base=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig8_error_gamma_PI.png')
    print('PI-small-grid: Iteration number when gamma is {:.3f}: {}'.format(gamma, result[-1]['Iteration']))
    print('PI-small-grid: Reward when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Reward']))
    print('PI-small-grid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('PI-small-grid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of PI in small Grid problem'
solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=0.999, eval_type=1, max_iter=iteration, skip_check=True)
solver.run()
policy=solver.policy
plot_policy_map(title, policy, desc, map_color(), number_to_directions())
plt.savefig('Fig8-1_map_visual_PI.png')
plt.close()

# 2. Big size Grid problem
np.random.seed(0)
size=20
random_map2 = generate_random_map(size=size)
map_env=FrozenLakeEnvExtend(desc=random_map2, re=[1.0, -1.0, -0.01], is_slippery=True)
desc=map_env.unwrapped.desc
P, R =OpenAI_MDPToolbox_modified(map_env).convert_PR()
iteration=10000
gammas=[0.7, 0.8, 0.9, 0.99, 0.999]

# 2-1. VI
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=False, run_stat_frequency=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Reward']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(9)
    plt.plot(a[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Reward of each gamma')
    plt.ylabel('Reward')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig9_reward_gamma_VI.png')
    plt.grid()
    plt.figure(10)
    plt.plot(b[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.savefig('Fig10_mean_V_gamma_VI.png')
    plt.grid()
    plt.figure(11)
    plt.plot(c[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig11_time_gamma_VI.png')
    plt.grid()
    plt.figure(12)
    plt.plot(d[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.yscale('log', base=10)
    plt.ylabel('Error(delta)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig12_error_gamma_VI.png')
    print('VI-Big-grid: Iteration number when gamma is {:.3f}: {}'.format(gamma, result[-1]['Iteration']))
    print('VI-Big-grid: Reward when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Reward']))
    print('VI-Big-grid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('VI-Big-grid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of VI in large Grid problem'
solver=hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=0.999, epsilon=epsilon, max_iter=iteration, skip_check=True, run_stat_frequency=True)
solver.run()
policy=solver.policy
plot_policy_map(title, policy, desc, map_color(), number_to_directions())
plt.savefig('Fig12-1_map_visual_VI.png')
plt.close()

# 2-2. PI
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver2=hiive.mdptoolbox.mdp.PolicyIterationModified(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=True)
    solver2.run()
    solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=solver2.iter, skip_check=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Reward']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(13)
    plt.plot(a[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Reward of each gamma')
    plt.ylabel('Reward')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig13_reward_gamma_PI.png')
    plt.grid()
    plt.figure(14)
    plt.plot(b[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig14_mean_V_gamma_PI.png')
    plt.grid()
    plt.figure(15)
    plt.plot(c[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig15_time_gamma_PI.png')
    plt.grid()
    plt.figure(16)
    plt.plot(d[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.yscale('log', base=10)
    plt.ylabel('Error(delta)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig16_error_gamma_PI.png')
    print('PI-Big-grid: Iteration number when gamma is {:.3f}: {}'.format(gamma, result[-1]['Iteration']))
    print('PI-Big-grid: Reward when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Reward']))
    print('PI-Big-grid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('PI-Big-grid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of PI in large Grid problem'
solver2 = hiive.mdptoolbox.mdp.PolicyIterationModified(P, R, gamma=0.999, epsilon=epsilon, max_iter=iteration,skip_check=True)
solver2.run()
solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=0.999, eval_type=1, max_iter=solver2.iter, skip_check=True)
solver.run()
policy=solver.policy
plot_policy_map(title, policy, desc, map_color(), number_to_directions())
plt.savefig('Fig16-1_map_visual_PI.png')
plt.close()

# 3. Small size Non-Grid problem
s=25
p=0.1
P, R=hiive.mdptoolbox.example.forest(S=s, p=p, r1=4, r2=2)

# 3-1. VI
iteration=10000
gamma=0.999
a=np.zeros(len(epsilons))
b=np.zeros(len(epsilons))
for i, epsilon in enumerate(epsilons):
    solver = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=gamma, epsilon=epsilon, max_iter=iteration, skip_check=True, run_stat_frequency=True)
    result = solver.run()
    a[i] = result[-1]['Reward']
    b[i] = result[-1]['Time']
    print('VI-small-nongrid(epsilon): Reward when epsilon is {}: {}'.format(epsilon, result[-1]['Reward']))
    print('VI-small-nongrid(epsilon): Time when epsilon is {}: {}'.format(epsilon, result[-1]['Time']))
plt.figure(0-4)
plt.grid()
x=['1e-5', '1e-7', '1e-10', '1e-12', '1e-15']
plt.plot(x, a, marker='o')
plt.title('Reward of each epsilon')
plt.ylabel('Reward')
plt.xlabel('Epsilon Value')
plt.tight_layout()
plt.savefig('Fig0-4_reward_epsilon_VI.png')
plt.close()

plt.figure(0-5)
plt.grid()
x=['1e-5', '1e-7', '1e-10', '1e-12', '1e-15']
plt.plot(x, b, marker='o')
plt.title('Time of each epsilon')
plt.ylabel('Time(sec)')
plt.xlabel('Epsilon Value')
plt.tight_layout()
plt.savefig('Fig0-5_time_epsilon_VI.png')
plt.close()

epsilon=1e-15
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=True, run_stat_frequency=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Reward']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(17)
    plt.plot(a[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Reward of each gamma')
    plt.ylabel('Reward')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig17_reward_gamma_VI.png')
    plt.grid()
    plt.figure(18)
    plt.plot(b[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig18_mean_V_gamma_VI.png')
    plt.grid()
    plt.figure(19)
    plt.plot(c[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig19_time_gamma_VI.png')
    plt.grid()
    plt.figure(20)
    plt.plot(d[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.yscale('log', base=10)
    plt.ylabel('Error(delta)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig20_error_gamma_VI.png')
    print('VI-small-nongrid: Iteration number when gamma is {:.3f}: {}'.format(gamma, result[-1]['Iteration']))
    print('VI-small-nongrid: Reward when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Reward']))
    print('VI-small-nongrid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('VI-small-nongrid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of VI in small Non-grid problem'
solver=hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=0.999, epsilon=epsilon, max_iter=iteration, skip_check=True, run_stat_frequency=True)
solver.run()
policy=solver.policy
plot_forest_policy_map(title, policy, s, map_color2(), number_to_action())
plt.savefig('Fig20-1_map_visual_VI.png')
plt.close()

# 3-2. PI
a=np.zeros(len(epsilons))
b=np.zeros(len(epsilons))
for i, epsilon in enumerate(epsilons):
    solver2=hiive.mdptoolbox.mdp.PolicyIterationModified(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=True)
    solver2.run()
    solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=solver2.iter, skip_check=True)
    #solver = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=iteration, skip_check=True)
    result = solver.run()
    a[i] = result[-1]['Reward']
    b[i] = result[-1]['Time']
    print('PI-small-nongrid(epsilon): Reward when epsilon is {}: {}'.format(epsilon, result[-1]['Reward']))
    print('PI-small-nongrid(epsilon): Time when epsilon is {}: {}'.format(epsilon, result[-1]['Time']))
plt.figure(0-6)
plt.grid()
x=['1e-5', '1e-7', '1e-10', '1e-12', '1e-15']
plt.plot(x, a, marker='o')
plt.title('Reward of each epsilon')
plt.ylabel('Reward')
plt.xlabel('Epsilon Value')
plt.tight_layout()
plt.savefig('Fig0-6_reward_epsilon_PI.png')
plt.close()

plt.figure(0-7)
plt.grid()
x=['1e-5', '1e-7', '1e-10', '1e-12', '1e-15']
plt.plot(x, b, marker='o')
plt.title('Time of each epsilon')
plt.ylabel('Time(sec)')
plt.xlabel('Epsilon Value')
plt.legend()
plt.tight_layout()
plt.savefig('Fig0-7_time_epsilon_PI.png')
plt.close()

epsilon=1e-15
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver2=hiive.mdptoolbox.mdp.PolicyIterationModified(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=True)
    solver2.run()
    solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=solver2.iter, skip_check=True)
    #solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=iteration, skip_check=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Reward']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(21)
    plt.plot(a[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Reward of each gamma')
    plt.ylabel('Reward')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig21_reward_gamma_PI.png')
    plt.grid()
    plt.figure(22)
    plt.plot(b[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig22_mean_V_gamma_PI.png')
    plt.grid()
    plt.figure(23)
    plt.plot(c[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig23_time_gamma_PI.png')
    plt.grid()
    plt.figure(24)
    plt.plot(d[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.yscale('log', base=10)
    plt.ylabel('Error(delta)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig24_error_gamma_PI.png')
    print('PI-small-nongrid: Iteration number when gamma is {:.3f}: {}'.format(gamma, result[-1]['Iteration']))
    print('PI-small-nongrid: Reward when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Reward']))
    print('PI-small-nongrid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('PI-small-nongrid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of PI in small Non-grid problem'
solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=0.999, eval_type=1, max_iter=iteration, skip_check=True)
solver.run()
policy=solver.policy
plot_forest_policy_map(title, policy, s, map_color2(), number_to_action())
plt.savefig('Fig24-1_map_visual_PI.png')
plt.close()

# 4. Big size Non-Grid problem
s=400
p=0.1
P, R=hiive.mdptoolbox.example.forest(S=s, p=p, r1=4, r2=2)
iteration=10000
# 4-1. VI
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=True, run_stat_frequency=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Reward']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(25)
    plt.plot(a[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Reward of each gamma')
    plt.ylabel('Reward')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig25_reward_gamma_VI.png')
    plt.grid()
    plt.figure(26)
    plt.plot(b[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig26_mean_V_gamma_VI.png')
    plt.grid()
    plt.figure(27)
    plt.plot(c[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.savefig('Fig27_time_gamma_VI.png')
    plt.grid()
    plt.figure(28)
    plt.plot(d[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.yscale('log', base=10)
    plt.ylabel('Error(delta)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig28_error_gamma_VI.png')
    print('VI-Big-nongrid: Iteration number when gamma is {:.3f}: {}'.format(gamma, result[-1]['Iteration']))
    print('VI-Big-nongrid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('VI-Big-nongrid: Reward when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Reward']))
    print('VI-Big-nongrid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of VI in large Non-grid problem'
solver=hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=0.99, epsilon=epsilon, max_iter=iteration, skip_check=True, run_stat_frequency=True)
solver.run()
policy=solver.policy
plot_forest_policy_map(title, policy, s, map_color2(), number_to_action())
plt.savefig('Fig28-1_map_visual_VI.png')
plt.close()

# 4-2. PI
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver2=hiive.mdptoolbox.mdp.PolicyIterationModified(P, R, gamma, epsilon=epsilon, max_iter=iteration, skip_check=True)
    solver2.run()
    solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma, eval_type=1, max_iter=solver2.iter, skip_check=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Reward']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(29)
    plt.plot(a[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Reward of each gamma')
    plt.ylabel('Reward')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig29_reward_gamma_PI.png')
    plt.grid()
    plt.figure(30)
    plt.plot(b[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig30_mean_V_gamma_PI.png')
    plt.grid()
    plt.figure(31)
    plt.plot(c[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig31_time_gamma_PI.png')
    plt.grid()
    plt.figure(32)
    plt.plot(d[i, 0:result[-1]['Iteration']], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.yscale('log', base=10)
    plt.ylabel('Error(delta)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig32_error_gamma_PI.png')
    print('PI-Big-nongrid: Iteration number when gamma is {:.3f}: {}'.format(gamma, result[-1]['Iteration']))
    print('PI-Big-nongrid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('PI-Big-nongrid: Reward when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Reward']))
    print('PI-Big-nongrid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of PI in large Non-grid problem'
solver2 = hiive.mdptoolbox.mdp.PolicyIterationModified(P, R, gamma=0.999, epsilon=epsilon, max_iter=iteration,skip_check=True)
solver2.run()
solver=hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=0.999, eval_type=1, max_iter=iteration, skip_check=True)
solver.run()
policy=solver.policy
plot_forest_policy_map(title, policy, s, map_color2(), number_to_action())
plt.savefig('Fig32-1_map_visual_PI.png')
plt.close()

# 5. Q-Learning for small grid
gammas=[0.7, 0.8, 0.9, 0.99, 0.999]
iteration=10000
np.random.seed(0)
size=5
random_map1 = generate_random_map(size=size)
map_env=FrozenLakeEnvExtend(desc=random_map1, re=[1.0, -1.0, -0.01], is_slippery=True)
desc=map_env.unwrapped.desc
P, R =OpenAI_MDPToolbox_modified(map_env).convert_PR()
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=1.0, alpha_decay=0.999, alpha_min=0.2, n_iter=iteration, skip_check=True, run_stat_frequency=True)  # alpha_min from 0.1 to 0.5
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Max V']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(40)
    plt.plot(a[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Max V of each gamma')
    plt.ylabel('Max V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig40_reward_gamma_Q.png')
    plt.grid()
    plt.figure(41)
    plt.plot(b[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig41_mean_V_gamma_Q.png')
    plt.grid()
    plt.figure(42)
    plt.plot(c[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig42_time_gamma_Q.png')
    plt.grid()
    plt.figure(43)
    plt.plot(d[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.yscale('log', base=10)
    plt.xlabel('Number of Iteration')
    plt.tight_layout()
    plt.legend()
    plt.savefig('Fig43_error_gamma_Q.png')
    print('Q-small-grid(1.0 alpha): Max V when gamma is {:.3f}: {}'.format(gamma, result[-1]['Max V']))
    print('Q-small-grid(1.0 alpha): Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('Q-small-grid(1.0 alpha): Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of Q-learning in small Grid problem'
gamma=0.999
solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=1.0, alpha_decay=0.999, alpha_min=0.2, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min is set to 0.2
solver.run()
policy=solver.policy
plot_policy_map(title, policy, desc, map_color(), number_to_directions())
plt.savefig('Fig43-1_map_visual_Q.png')
plt.close()

a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=0.8, alpha_decay=0.999, alpha_min=0.2, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min from 0.1 to 0.5
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Max V']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(44)
    plt.plot(a[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Max V of each gamma')
    plt.ylabel('Max V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig44_reward_gamma_Q.png')
    plt.grid()
    plt.figure(45)
    plt.plot(b[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig45_mean_V_gamma_Q.png')
    plt.grid()
    plt.figure(46)
    plt.plot(c[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig46_time_gamma_Q.png')
    plt.grid()
    plt.figure(47)
    plt.plot(d[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.yscale('log', base=10)
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig47_error_gamma_Q.png')
    print('Q-small-grid(0.8 alpha): Max V when gamma is {:.3f}: {}'.format(gamma, result[-1]['Max V']))
    print('Q-small-grid(0.8 alpha): Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('Q-small-grid(0.8 alpha): Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))

a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=0.6, alpha_decay=0.999, alpha_min=0.2, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min from 0.1 to 0.5
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Max V']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(48)
    plt.plot(a[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Max V of each gamma')
    plt.ylabel('Max V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig48_reward_gamma_Q.png')
    plt.grid()
    plt.figure(49)
    plt.plot(b[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig49_mean_V_gamma_Q.png')
    plt.grid()
    plt.figure(50)
    plt.plot(c[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig50_time_gamma_Q.png')
    plt.grid()
    plt.figure(51)
    plt.plot(d[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.yscale('log', base=10)
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig51_error_gamma_Q.png')
    print('Q-small-grid(0.6 alpha): Max V when gamma is {:.3f}: {}'.format(gamma, result[-1]['Max V']))
    print('Q-small-grid(0.6 alpha): Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('Q-small-grid(0.6 alpha): Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

#6. Q-Learning for large grid
iteration=1000000
np.random.seed(0)
size=20
random_map1 = generate_random_map(size=size)
map_env=FrozenLakeEnvExtend(desc=random_map1, re=[1.0, -1.0, -0.01], is_slippery=True)
desc=map_env.unwrapped.desc
P, R =OpenAI_MDPToolbox_modified(map_env).convert_PR()

a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.001, alpha=1.0, alpha_decay=0.999, alpha_min=0.2, n_iter=iteration, skip_check=True, run_stat_frequency=True)  # alpha_min from 0.1 to 0.5
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Max V']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(56)
    plt.plot(a[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Max V of each gamma')
    plt.ylabel('Max V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig56_reward_gamma_Q.png')
    plt.grid()
    plt.figure(57)
    plt.plot(b[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig57_mean_V_gamma_Q.png')
    plt.grid()
    plt.figure(58)
    plt.plot(c[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig58_time_gamma_Q.png')
    plt.grid()
    plt.figure(59)
    plt.plot(d[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.yscale('log', base=10)
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig59_error_gamma_Q.png')
    print('Q-Big-grid: Max V when gamma is {:.3f}: {}'.format(gamma, result[-1]['Max V']))
    print('Q-Big-grid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('Q-Big-grid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of Q-learning in large Grid problem'
gamma=0.999
solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=1.0, alpha_decay=0.999, alpha_min=0.2, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min is set to 0.2
solver.run()
policy=solver.policy
plot_policy_map(title, policy, desc, map_color(), number_to_directions())
plt.savefig('Fig59-1_map_visual_Q.png')
plt.close()

#7. Small size Non-Grid problem
s=25
p=0.1
P, R=hiive.mdptoolbox.example.forest(S=s, p=p, r1=4, r2=2)
iteration=10000
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=1.0, alpha_decay=0.999, alpha_min=0.5, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min from 0.1 to 0.5
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Max V']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(60)
    plt.plot(a[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Max V of each gamma')
    plt.ylabel('Max V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig60_reward_gamma_Q.png')
    plt.grid()
    plt.figure(61)
    plt.plot(b[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig61_mean_V_gamma_Q.png')
    plt.grid()
    plt.figure(62)
    plt.plot(c[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig62_time_gamma_Q.png')
    plt.grid()
    plt.figure(63)
    plt.plot(d[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.yscale('log', base=10)
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig63_error_gamma_Q.png')
    print('Q-small-Nongrid(1.0 alpha): Max V when gamma is {:.3f}: {}'.format(gamma, result[-1]['Max V']))
    print('Q-small-Nongrid(1.0 alpha): Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('Q-small-Nongrid(1.0 alpha): Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
    print('policy: ', solver.policy)
plt.close()

a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=0.8, alpha_decay=0.999, alpha_min=0.5, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min from 0.1 to 0.5
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Max V']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(64)
    plt.plot(a[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Max V of each gamma')
    plt.ylabel('Max V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig64_reward_gamma_Q.png')
    plt.grid()
    plt.figure(65)
    plt.plot(b[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig65_mean_V_gamma_Q.png')
    plt.grid()
    plt.figure(66)
    plt.plot(c[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig66_time_gamma_Q.png')
    plt.grid()
    plt.figure(67)
    plt.plot(d[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.yscale('log', base=10)
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig67_error_gamma_Q.png')
    print('Q-small-Nongrid(0.8 alpha): Max V when gamma is {:.3f}: {}'.format(gamma, result[-1]['Max V']))
    print('Q-small-Nongrid(0.8 alpha): Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('Q-small-Nongrid(0.8 alpha): Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=0.6, alpha_decay=0.999, alpha_min=0.5, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min from 0.1 to 0.5
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Max V']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(68)
    plt.plot(a[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Max V of each gamma')
    plt.ylabel('Max V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig68_reward_gamma_Q.png')
    plt.grid()
    plt.figure(69)
    plt.plot(b[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig69_mean_V_gamma_Q.png')
    plt.grid()
    plt.figure(70)
    plt.plot(c[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig70_time_gamma_Q.png')
    plt.grid()
    plt.figure(71)
    plt.plot(d[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.yscale('log', base=10)
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig71_error_gamma_Q.png')
    print('Q-small-Nongrid(0.6 alpha): Max V when gamma is {:.3f}: {}'.format(gamma, result[-1]['Max V']))
    print('Q-small-Nongrid(0.6 alpha): Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('Q-small-Nongrid(0.6 alpha): Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
plt.close()

title='Policy of Q-learning in small Non-grid problem'
solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=1.0, alpha_decay=0.999, alpha_min=0.5, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min is set to 0.5
solver.run()
policy=solver.policy
plot_forest_policy_map(title, policy, s, map_color2(), number_to_action())
plt.savefig('Fig75-1_map_visual_VI.png')
plt.close()

#8. Large size Non-Grid problem
s=400
p=0.1
P, R=hiive.mdptoolbox.example.forest(S=s, p=p, r1=4, r2=2)
iteration=1000000
a=np.zeros((len(gammas), iteration))
b=np.zeros((len(gammas), iteration))
c=np.zeros((len(gammas), iteration))
d=np.zeros((len(gammas), iteration))
for i, gamma in enumerate(gammas):
    solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=1.0, alpha_decay=0.999, alpha_min=0.5, n_iter=iteration, skip_check=True, run_stat_frequency=True)
    result=solver.run()
    for j, k in enumerate(result):
        a[i, j] = k['Max V']
        b[i, j] = k['Mean V']
        c[i, j] = k['Time']
        d[i, j] = k['Error']
    plt.grid()
    plt.figure(76)
    plt.plot(a[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Max V of each gamma')
    plt.ylabel('Max V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.savefig('Fig76_reward_gamma_Q.png')
    plt.grid()
    plt.figure(77)
    plt.plot(b[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Mean V of each gamma')
    plt.ylabel('Mean V')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig77_mean_V_gamma_Q.png')
    plt.grid()
    plt.figure(78)
    plt.plot(c[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Time of each gamma')
    plt.ylabel('Time(sec)')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig78_time_gamma_Q.png')
    plt.grid()
    plt.figure(79)
    plt.plot(d[i, :], label='Gamma: {:.3f}'.format(gamma))
    plt.grid()
    plt.title('Error of each gamma')
    plt.ylabel('Error(delta)')
    plt.yscale('log', base=10)
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig79_error_gamma_Q.png')
    print('Q-Big-Nongrid: Max V when gamma is {:.3f}: {}'.format(gamma, result[-1]['Max V']))
    print('Q-Big-Nongrid: Mean V when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Mean V']))
    print('Q-Big-Nongrid: Time when gamma is {:.3f}: {:.4f}'.format(gamma, result[-1]['Time']))
    print('policy: ', solver.policy)
plt.close()

title='Policy of Q-learning in large Non-grid problem'
solver=hiive.mdptoolbox.mdp.QLearning(P, R, gamma=0.999, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001, alpha=1.0, alpha_decay=0.999, alpha_min=0.5, n_iter=iteration, skip_check=True, run_stat_frequency=True) # alpha_min is set to 0.5
solver.run()
policy=solver.policy
plot_forest_policy_map(title, policy, s, map_color2(), number_to_action())
plt.savefig('Fig79-1_map_visual_VI.png')
plt.close()
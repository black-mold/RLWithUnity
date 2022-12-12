# 라이브러리 불러오기
import numpy as np
import random
import copy
import datetime
import platform
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter
from collections import deque

from mlagents_envs.environment import UnityEnvironment, ActionTuple


vector_size = 8
action_size = 2

# DQN을 위한 파라미터 값 세팅 
load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 1000
discount_factor = 0.9
learning_rate = 0.00025

run_step = 50000 if train_mode else 0

target_update_step = 500

print_interval = 10
save_interval = 200

epsilon_eval = 0.05
epsilon_init = 0.5 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = run_step * 0.8
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0.

device = 'cuda'
save_path = './results/dqn_python_api'



# DQN 클래스 -> Deep Q Network 정의 
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의 
class DQNAgent:
    def __init__(self):
        self.network =  Qnet().to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)


    # Epsilon greedy 기법에 따라 행동 결정 
    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval

        # 랜덤하게 행동 결정
        if epsilon > random.random():  
            action = np.random.uniform(low=-1.0, high=1.0, size=(state.shape[0], action_size)
        )
        # 네트워크 연산에 따라 행동 결정
        else:
            action = self.network(torch.FloatTensor(state).to(device))
            action = action.data.cpu().numpy()
        return action

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done       = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])

        eye = torch.eye(action_size).to(device)

        action = torch.argmax(action, axis=-1, keepdim=True)
        one_hot_action = eye[action.view(-1).long()]

        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            next_q = self.target_network(next_state)
            target_q = reward + next_q.max(1, keepdims=True).values * ((1 - done) * discount_factor)

        loss = F.smooth_l1_loss(q, target_q)
        # print(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 엡실론 감소
        self.epsilon = max(epsilon_min, self.epsilon - eplsilon_delta)
        # print(self.epsilon)

        return loss.item()

    # 타겟 네트워크 업데이트
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    # 네트워크 모델 저장 
    def save_model(self, eps):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+f'/ckpt_{eps}')

    # 학습 기록 
    def write_summray(self, score, loss, epsilon, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/epsilon", epsilon, step)


if __name__ == '__main__':
    # define RL environment
    env = UnityEnvironment(file_name = './project_unity_hub/ball3d/ball3d_env/ball3d')

    # 환경 초기화
    env.reset()

    # behavior 관련 정보 저장
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    # DQN 클래스를 agent로 정의
    agent = DQNAgent()

    losses, scores, episode, score = [], [], 0, 0

    # episode 진행
    for ep in range(100000000):

        # 환경 초기화
        env.reset()

        # 에이전트가 행동을 요청한 상태인지, 마지막 상태인지 확인
        decision_steps, terminal_steps = env.get_steps(behavior_name) # env.get_step : 각 스텝에서 에이전트의 상태, 행동, 보상 등의 정보를 반환한다

        # 환경 관련 변수 초기화
        done = False # 에피소드가 마무리 됐는지 판단.
        ep_rewards = 0


        ### 이제부터 에피소드가 본격적으로 진행됨
        while not done:

            # tracked agent 지정
            if (len(decision_steps)) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # state, 액션 : epsilon-greedy            
            state = decision_steps.obs[0] # state shape : (12, 8)
            action = agent.get_action(state, train_mode)
            
            action_tuple = ActionTuple()
            action_tuple.add_continuous(action)

            # 엑션 설정            
            env.set_actions(behavior_name, action_tuple)


            # 실제 액션 수행
            env.step()

            # 스텝 종료 후 에이전트의 정보 (보상, 상태 등) 취득
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # terminal Check
            if (len(terminal_steps.agent_id) > 0) :
                done = True
                # reward
                reward = terminal_steps.reward 
                # next_state
                next_state = terminal_steps.obs[0]
            
            # not terminal
            else :
                done = False
                reward = decision_steps.reward
                next_state = decision_steps.obs[0]

            score += reward[0]
            ep_rewards += reward[0]

            if train_mode:
                agent.append_sample(state[0], action[0], reward[0], next_state[0], [done])

            if train_mode and ep > batch_size:
                # 학습 수행
                loss = agent.train_model()
                losses.append(loss)

            # 타겟 네트워크 업데이트 
            if ep % target_update_step == 0:
                agent.update_target()

            if done:
                episode +=1
                scores.append(score)
                score = 0

                # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
                if episode % print_interval == 0:
                    mean_score = np.mean(scores)
                    mean_loss = np.mean(losses)
                    agent.write_summray(mean_score, mean_loss, agent.epsilon, ep)
                    losses, scores = [], []

                    print(f"{episode} Episode / Step: {ep} / Score: {mean_score:.2f} / " +\
                        f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

                # 네트워크 모델 저장 
                if train_mode and episode % save_interval == 0:
                    agent.save_model(episode)


        
        # 한 에피소드가 종료되고, 추적중인 에이전트에 대해서 해당 에피소드에서의 보상 출력
        # print(f"total reward for ep {ep} is {ep_rewards}")

    # 환경 종료
    env.close()
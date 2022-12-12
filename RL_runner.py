from mlagents_envs.environment import UnityEnvironment
import torch.nn as nn
import torch






if __name__ == '__main__':
    # define RL environment
    env = UnityEnvironment(file_name = './project_unity_hub/ball3d/ball3d_env/ball3d')

    # 환경 초기화
    env.reset()

    # behavior 관련 정보 저장
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]


    # episode 진행
    for ep in range(1000):

        # 환경 초기화
        env.reset()

        # 에이전트가 행동을 요청한 상태인지, 마지막 상태인지 확인
        ## 해당 정보들이 다음 행동을 요청한 스텝이라면, 즉 아직 환경의 에피소드가 끝나지 않은 상태로 스텝이 진행중이라면 decision_step에 저장된다.
        ## 만약, 에피소드가 끝난 마지막 스텝이라면 time_step에 정보들이 저장되고, decision_step에는 다음 에피소드의 첫 스텝 정보가 저장된다.
        decision_steps, terminal_steps = env.get_steps(behavior_name) # env.get_step : 각 스텝에서 에이전트의 상태, 행동, 보상 등의 정보를 반환한다.


        # 환경 관련 변수 초기화
        # 한 에이전트를 기준으로 로그를 출력
        tracked_agent = -1 # 추적할 에이전트의 아이디를 저장하는데 , 총 12개의 에이전트가 있지만 하나만 저장
        done = False # 에피소드가 마무리 됐는지 판단.
        ep_rewards = 0 # 보상의 합을 저장할 변수


        ### 이제부터 에피소드가 본격적으로 진행됨
        while not done:

            # tracked agent 지정
            if tracked_agent == -1 and (len(decision_steps)) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # 랜덤 액션 결정
            # print(len(decision_steps)) # 12
            action = spec.action_spec.random_action(len(decision_steps))

            # 엑션 설정
            env.set_actions(behavior_name, action)

            # 실제 액션 수행
            env.step()

            # 스텝 종료 후 에이전트의 정보 (보상, 상태 등) 취득
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # 추적중인 에이전트가 행동이 가능한 상태와 종료 상태일 때를 구분하여 보상 저장

            if tracked_agent in decision_steps:
                ep_rewards += decision_steps[tracked_agent].reward

            # print(tracked_agent in terminal_steps)
            if tracked_agent in terminal_steps:
                ep_rewards += terminal_steps[tracked_agent].reward
                done = True
        
        # 한 에피소드가 종료되고, 추적중인 에이전트에 대해서 해당 에피소드에서의 보상 출력

        print(f"total reward for ep {ep} is {ep_rewards}")

    # 환경 종료
    env.close()
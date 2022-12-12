# Unity ML-Agents Toolkit

[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://github.com/Unity-Technologies/ml-agents/tree/release_20_docs/docs/)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](../LICENSE.md)

([latest release](https://github.com/Unity-Technologies/ml-agents/releases/tag/latest_release))
([all releases](https://github.com/Unity-Technologies/ml-agents/releases))

**The Unity Machine Learning Agents Toolkit** (ML-Agents) is an open-source
project that enables games and simulations to serve as environments for
training intelligent agents. We provide implementations (based on PyTorch)
of state-of-the-art algorithms to enable game developers and hobbyists to easily
train intelligent agents for 2D, 3D and VR/AR games. Researchers can also use the
provided simple-to-use Python API to train Agents using reinforcement learning,
imitation learning, neuroevolution, or any other methods. These trained agents can be
used for multiple purposes, including controlling NPC behavior (in a variety of
settings such as multi-agent and adversarial), automated testing of game builds
and evaluating different game design decisions pre-release. The ML-Agents
Toolkit is mutually beneficial for both game developers and AI researchers as it
provides a central platform where advances in AI can be evaluated on Unity’s
rich environments and then made accessible to the wider research and game
developer communities.

### 1. ml-agent실행

1. 해당 Repository clone
2. 프로젝트 설치 후 강화학습을 위한 유니티 환경 Build(PPT 참고)
3. Build한 유니티 환경에 대한 Model 학습 진행
```{python}
# PPO(Proximal Policy Optimization)
mlagents-learn {trainer_path} --env={env_path}/{build_name} --run-id={run_id}
```
-- PPO algorithm Hyperparameter에 관한 사항은 ./config/ppo/3DBall.yaml 

### 2. Python-API DQN 실행
```
# DQN 학습
python RL_Trainer_dqn.py

# DQN 학습 결과 Test
python RL_Tester_dqn.py
```


### References
1. folk [ml-agents @ release_17](https://github.com/Unity-Technologies/ml-agents/releases/tag/release_17)

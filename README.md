# Continuous-Actor-Critic-Learning-Automaton

본 프로젝트는 한국항공대학교 Reinforcement Learning과목의 프로젝트 제출을 목적으로 만들어짐.

<div style="display:flex;">
<img width="30%" src="https://github.com/nninept/Continuous-Actor-Critic-Learning-Automaton/blob/master/src/gif/halfcheetah.gif"/>
<img width="30%" src="https://github.com/nninept/Continuous-Actor-Critic-Learning-Automaton/blob/master/src/gif/pendulum.gif"/>
<img width="30%" src="https://github.com/nninept/Continuous-Actor-Critic-Learning-Automaton/blob/master/src/gif/Ant.gif"/>
<img width="30%" src="https://github.com/nninept/Continuous-Actor-Critic-Learning-Automaton/blob/master/src/gif/humanoid.gif"/>
<img width="30%" src="https://github.com/nninept/Continuous-Actor-Critic-Learning-Automaton/blob/master/src/gif/hopper.gif"/>
</div>
  
### Algorithm

Continuous Actor-Critic Learning Automaton (CARCLA) 모델을 이용하여 Mujoco환경을 학슴.
논문의 경우 https://www.researchgate.net/publication/4249966_Reinforcement_Learning_in_Continuous_Action_Spaces 을 참고함.  
이때, 초기 논문의 알고리즘을 그대로 사용할 경우, 학습의 속도가 현저히 느리며 성능도 높은 편이 아닌 것을 확인.

이를 해결하기 위해 [CARCLA에 Replay Buffer를 활용한 논문](https://proceedings.mlr.press/v101/wang19a.html) 을 참고하여 Replay Buffer를 사용함.
이전에 비해 학습 속도가 증가한 것을 알 수 있었다.

### Environment

OpenAI의 gym에서 제공하는 Mujoco환경들 중 5가지를 골라 학습함.
[HalfCheetah](https://www.gymlibrary.ml/environments/mujoco/half_cheetah/), [InvertedPendulum](https://www.gymlibrary.ml/environments/mujoco/inverted_pendulum/), 
[Ant](https://www.gymlibrary.ml/environments/mujoco/ant/), [Humanoid](https://www.gymlibrary.ml/environments/mujoco/humanoid/), [Hopper](https://www.gymlibrary.ml/environments/mujoco/hopper/)
환경들을 사용하였다.

### Performance 

![performance](https://github.com/nninept/Continuous-Actor-Critic-Learning-Automaton/blob/master/src/image/performance-graph.png)
HalfCheetah, InvertedPendulum에서는 준수한 성능을 보였다. Ant의 경우 훈련동안 이상한 모습을 보였는데, 한 에피소드 동안 축적된 Reward가 대략 800선에서 시작하여 -200대까지 점차 
작아지는 모습을 보였다. 앞선 두 환경에서 학습이 정상적으로 된 것과, 학습이 진행되면서 리워드의 증감이 일관적이었다는 점에서, 정상적으로 훈련이 된 것으로 판단되나, 그 성능이 좋아지지는
않았다는 점에서 추가적인 관찰이 필요할 것으로 보인다. Humanoid와 Hopper의 경우, Reward의 증감에 있어 변화가 없는 것으로 보아, 훈련한 네트워크 구조로는 훈련이 힘든 것으로 보인다.
네트워크의 크기를 키우거나, state를 input하는 과정에서 추가적인 작업이 필요할 것으로 보인다. 모든 환경들을 학습할때 동일한 네트워크 구조와 Hyper Parameter를 이용했다.

### Setting

##### Network

```python
self.network = nn.Sequential(
            nn.Linear(observation_space[0], 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256,action_space[0]),
            nn.Tanh()
        ).to(device)
```
Action 네트워크와 Value 네트워크의 구조는 동일한 것으로 사용하였다. 이때, Action 네트워크와 Value 네트워크의 Weight들을 공유하는 방식을 사용할 경우, 초반 학습은 빠르게 진행이
되었지만, 이후 훈련이 진행되면서 성능이 저조해지는 현상이 있었다. 결과적으로는 Action 네트워크와 Value 네트워크들을 따로 두는 것이 최종 성능에서 더 높은 모습을 보였다.  
두 네트워크 모두 input layer의 크기는 환경에 따라 달라지도록 설정했으며, output layer의 경우 Action 네트워크는 action spcae의 크기만큼, Value 네트워크는 1로 두었다.


##### Input State

모든 Input들은 수치형이며, 별도의 Scaling 없이 네트워크로 입력되었다. 공식 문서에 따르면 모든 환경들의 State 값들은 [-inf, inf]로, 이론상 상당히 값들이 큰 것을 알 수 있다. 허나 실제
훈련시에는 이렇게 큰 값이 나오지는 않는것으로 확인하였다.  

RNN을 이용하여 이전의 여러 State들을 같이 넣어주는 구조로 네트워크를 작성해보기도 했지만, 실제 훈련시 그리 높은 성능을 못 내는 것으로 확인하였다. Humanoid와 같이
시퀀스의 정보가 중요한 경우에는 이득이 될 것으로 보인다. 허나 앞서 언급했듯 해당 환경들은 상당한 크기의 구조를 필요로 하는 것으로 보이며, 이 경우 학습의 안정성이 상당히 떨어지는
것으로 보인다.

##### Action

```python
    def select_action(self, state):
        action = self.forward(state)
        policy = torch.normal(action.detach(), 0.1)
        policy = torch.clamp(policy, max=self.action_range, min=self.action_range*(-1))
        return policy
```

Action 도출은 CARCLA논문을 참고하였다. Action시 Exploration을 위하여 N(a, v)의 값을 이용하는데, 이때 mean이 action이므로 v가 크면 Exploration이 증가한다.
하지만 v가 커지면 네트워크에서 도출된 Action의 변동이 커져 학습이 어려워진다. 본 프로젝트에서는 0.1으로 고정하였다.

Action 네트워크의 경우 Tanh를 적용시켰는데, Mujoco 환경들의 action space가 모두 0을 기준으로 최솟값과 최댓값의 절대값이 같은 것을 확인했다. 이에 따라 Tanh를 나온 값에 action
space 최대값을 곱해주어 range를 맞추었다.

##### Other Settings

Polcy Gradient의 경우, 한 네트워크로만 학습을 시키면 학습이 불안정한 것을 찾아볼 수 있다. 이를 해결하기 위해 DDPG에서 적용되었던 Soft-Update를 사용하여, Target Network와
Behavior Network를 따로 두고 점진적으로 학습을 진행하였다. 

Replay Buffer를 사용하였기 때문에, 학습단계(Step)마다 훈련을 진행하였는데, 1번만 학습시키는 것보단 여러번 학습시키는 것이 낫다 판단하여, 매 단계마다 Replay Buffer에서 15번 Sampling
해 학습하였다.

Optimizer는 RMSProp을 사용하였다. 강화학습의 경우 네트워크의 학습 변동성이 크다는 점을 고려하여, RMSProp이 Momentum이 들어가는 Adam보다는 더 유리할 것이라는 가설때문이었다.
이에 대한 비교분석은 훈련 시간으로 인해 확인해보지 못하였다.

모든 학습은 연구실의 컴퓨터를 사용하였다. GPU는 3080ti, CPU는 Ryzen 5600X이다. 학교 서버를 사용하려 했으나, mujoco-py를 사용하기 위해 설치해야하는 unbuntu package들이 
몇가지 충돌을 일으켜 사용하지 못했다. 

##### Hyper Parameter

Learning Rate는 0.001로 고정하였다. Scheduler를 사용하지 않은 것은, 모든 환경들이 학습이 완료되는 시점이 제각각이기 때문이었다.  

Gamma는 0.9로 설정하였다.

Rpelay Buffer Sampling 개수는 32로 설정하였다.

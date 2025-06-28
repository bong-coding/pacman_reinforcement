# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for count in range(self.iterations):
            temp = util.Counter()
            # 모든 상태 순회
            for st in self.mdp.getStates():
                # 종료 상태는 값 0 유지
                if self.mdp.isTerminal(st):
                    temp[st] = 0
                else:
                    best_q = None
                    # 각 행동별 q값 계산
                    for act in self.mdp.getPossibleActions(st):
                        q_sum = 0
                        # 다음 상태 및 확률에 대해 누적
                        for nxt, pr in self.mdp.getTransitionStatesAndProbs(st, act):
                            rew = self.mdp.getReward(st, act, nxt)
                            q_sum = q_sum + pr * (rew + self.discount * self.values[nxt])
                        # bestq 초기화 및 비교 갱신
                        if best_q is None or q_sum > best_q:
                            best_q = q_sum
                    # 가능한 행동이 없으면 0, 있으면 최댓값
                    temp[st] = best_q if best_q is not None else 0
            # 계산된 값을 한꺼번에 적용
            self.values = temp
        

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
       # 시작할 때 q 값을 0으로 설정
        q_value = 0
        # 전이 목록 가져오기
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        # 각 전이에 대해 계산
        for trans in transitions:
            # 튜플에서 다음 상태와 확률 꺼내기
            next_state = trans[0]
            prob = trans[1]
            # 즉시 보상 
            reward = self.mdp.getReward(state, action, next_state)
            # 할인미래 가치 계산
            discounted_value = self.discount * self.values[next_state]
            # 누적 q 값에 더하기
            q_value = q_value + prob * (reward + discounted_value)
        # 최종 qv 반환
        return q_value
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
         # 터미널 상태일 경우 바로 None 반환
        if self.mdp.isTerminal(state):
            return None
        # 가능한 행동 리스트 가져오기
        actions = self.mdp.getPossibleActions(state)
        # 초기값 설정: 첫 행동을 임시로 최선의 행동으로 지정
        best_act = None
        # q 값을 초기화할 때 None 체크
        best_score = None
        # 각 행동에 대해 q 값 계산
        for act in actions:
            # 행동에 대한 q 값 얻기
            score = self.computeQValueFromValues(state, act)
            # best_score가 none 이거나 더 크면 업데이트
            if best_score is None or score > best_score:
                best_score = score
                best_act = act
        return best_act
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        # predecessors를 set으로 초기화
        preds = {}
        for s in states:
            preds[s] = set()
        # predecessors에 이전 상태 넣기
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            for a in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(s, a)
                for next_s, prob in transitions:
                    if prob > 0:
                        preds[next_s].add(s)

        # diff 계산해서 우선순위 큐에 update
        pq = util.PriorityQueue()
        for s in states:
            if self.mdp.isTerminal(s):
                continue
            best_q = None
            acts = self.mdp.getPossibleActions(s)
            for a in acts:
                q = self.computeQValueFromValues(s, a)
                if best_q is None or q > best_q:
                    best_q = q
            if best_q is None:
                best_q = 0
            diff = abs(self.values[s] - best_q)
            pq.update(s, -diff)

        # 반복 시작
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            # 터미널 아니면 value 업데이트
            if not self.mdp.isTerminal(s):
                best_q = None
                acts = self.mdp.getPossibleActions(s)
                for a in acts:
                    q = self.computeQValueFromValues(s, a)
                    if best_q is None or q > best_q:
                        best_q = q
                if best_q is None:
                    best_q = 0
                self.values[s] = best_q

            # 이전상태 마다 diff 계산해서 theta 넘으면 큐에 update
            for pre in preds[s]:
                if self.mdp.isTerminal(pre):
                    continue
                best_q_pre = None
                acts_pre = self.mdp.getPossibleActions(pre)
                for a in acts_pre:
                    q = self.computeQValueFromValues(pre, a)
                    if best_q_pre is None or q > best_q_pre:
                        best_q_pre = q
                if best_q_pre is None:
                    best_q_pre = 0
                diff_pre = abs(self.values[pre] - best_q_pre)
                if diff_pre > self.theta:
                    pq.update(pre, -diff_pre)

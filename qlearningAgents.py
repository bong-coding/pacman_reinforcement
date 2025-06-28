# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
         # qv 저장
        self.qValues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
         # util.raiseNotDefined()
        # s,a 반환 갱신된 적 없으면 기본값 0 반환
        pair = (state, action)
        if pair in self.qValues:
            return self.qValues[pair]
        return 0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        actions = self.getLegalActions(state)
        found_action = False
        for _ in actions:
            found_action = True
            break
        if not found_action:
            return 0.0

        # 최대 Q값 직접 찾기
        max_q = None
        for a in actions:
            q = self.getQValue(state, a)
            if max_q is None or q > max_q:
                max_q = q
        if max_q is None:
            return 0.0
        return max_q

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        actions = self.getLegalActions(state)
        is_action_exist = False
        for _ in actions:
            is_action_exist = True
            break
        if not is_action_exist:
            return None

        # 최대 Q값 찾기
        max_q = None
        for a in actions:
            q = self.getQValue(state, a)
            if max_q is None or q > max_q:
                max_q = q

        # 마지막 최대 Q값 action을 반환
        best_action = None
        for a in actions:
            if self.getQValue(state, a) == max_q:
                best_action = a
        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        pos_action = False
        for _ in legalActions:
            pos_action = True
            break
        if not pos_action:
            return None
        

        if util.flipCoin(self.epsilon):
            # 무작위 인덱스를 직접 구해서 legalActions 선택
            count = 0
            for _ in legalActions:
                count += 1
            rand = random.random()
            idx = 0
            reached = False
            for i in range(count):
                if not reached and rand < float(i + 1) / float(count):
                    idx = i
                    reached = True
            # 인덱스에 맞는 action 찾기
            j = 0
            for a in legalActions:
                if j == idx:
                    legalActions = a
                j += 1
        else:
            legalActions = self.computeActionFromQValues(state)
        return legalActions
        

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # 현재 q값 가져오기
        current_q = self.getQValue(state, action)

        # nextstate의 최대 q값을 직접 찾음
        next_q = self.getValue(nextState)

        # sample = r + discount * max(q(s',a'))
        sample = reward + self.discount * next_q

        # qv 갱tㅣㄴ
        new_q = (1 - self.alpha) * current_q + self.alpha * sample
        self.qValues[(state, action)] = new_q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        q_total = 0  # qv를 저장할 임시 변수
        for feature in features:
            feat_val = features[feature]    # 현재 feature의 값
            weight_val = self.weights[feature]  # 현재 feature의 가중치
            q_total = q_total + weight_val * feat_val  # qv 누적
        return q_total
        # util.raiseNotDefined()
        

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        current_q = self.getQValue(state, action)
        next_actions = self.getLegalActions(nextState)

        next_actions = self.getLegalActions(nextState)
        if not next_actions:  # 행동이 없으면 0
            next_q = 0.0
        else:
            next_q = float('-inf')
            for next_action in next_actions:
                q = self.getQValue(nextState, next_action)
                if q > next_q:
                    next_q = q

        # 오차 계산: (r + gamma * next_q) - current_q
        difference = (reward + self.discount * next_q) - current_q

        # feature별로 가중치 업데이트
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.weights[feature] = self.weights[feature] + self.alpha * difference * features[feature]
        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

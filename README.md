## 3 - 강화학습 (Value Iteration & Q-Learning)

 **Value Iteration**과 **Q-Learning**을 직접 구현하고, GridWorld 및 Pacman 환경에서 Agent가 최적의 정책을 학습함

---

## 📁 디렉토리 구조
```
reinforcement/
├── valueIterationAgents.py # Q1, Q4 구현
├── qlearningAgents.py # Q5~Q9 구현
├── analysis.py # Q2, Q3, Q7 매개변수 입력
├── mdp.py # MDP 정의
├── learningAgents.py # 기본 Agent 클래스
├── util.py # Counter 등 유틸리티
├── gridworld.py # GridWorld 환경 실행
├── crawler.py # 로봇 크롤러 환경
├── featureExtractors.py # 특징 추출기
├── autograder.py # 자동 채점기
├── test_cases/ # 테스트 케이스 디렉토리
└── 기타 그래픽 및 환경 파일들

```
----
```bash
python gridworld.py -a value -i 100
python gridworld.py -a q -k 100
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid 
```

---

## 구현파일

1. **Q1**  
   - 파일: `valueIterationAgents.py`  
   - 내용: Value Iteration 알고리즘 구현

2. **Q2~Q3**  
   - 파일: `analysis.py`  
   - 내용: BridgeGrid, DiscountGrid 실험용 파라미터 설정

3. **Q4**  
   - 파일: `valueIterationAgents.py`  
   - 내용: Prioritized Sweeping Value Iteration 구현 

4. **Q5~Q6**  
   - 파일: `qlearningAgents.py`  
   - 내용: Q-Learning 알고리즘 핵심 로직 (`getQValue`, `update`, `getAction` 등)

5. **Q7**  
   - 파일: `analysis.py`  
   - 내용: BridgeGrid 환경에서 epsilon, alpha 설정을 통한 실험

6. **Q8**  
   - 파일: `PacmanQAgent` *(기본 제공)*  
   - 내용: Pacman 환경에서 Q-Learning 수행

7. **Q9**  
   - 파일: `qlearningAgents.py`, `featureExtractors.py`  
   - 내용: 특징 기반 Approximate Q-Learning 구현 (가중치 학습 포함)

----
### 실행 명령어

```bash
# Value Iteration 결과 확인
python gridworld.py -a value -i 5

# Q-Learning 수동 학습
python gridworld.py -a q -k 5 -m

# Epsilon 테스트
python gridworld.py -a q -k 100 --noise 0.0 -e 0.3

# Pacman Q-Learning
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

# Approximate Q-Learning
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic

#자동 채점
python pacman.py autograder.py
```

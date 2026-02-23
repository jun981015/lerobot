# Lerobot Git 사용 가이드

## 전략 요약

- **서버 ↔ 로컬 동기화**: `origin/main` 단일 브랜치로 push/pull
- **공식 upstream 반영**: `git merge upstream/main` 으로 명시적 merge

## Remote 구성

| 이름 | URL | 용도 |
|------|-----|------|
| `origin` | https://github.com/jun981015/lerobot.git | 내 fork (서버↔로컬 중계) |
| `upstream` | https://github.com/huggingface/lerobot.git | 공식 huggingface repo |

```
huggingface/lerobot (upstream/main)
        ↓ git merge upstream/main (필요할 때만)
  jun981015/lerobot (origin/main)  ←→  로컬 main
        ↕ push/pull
   서버 (worker1)  ↔  로컬 PC
```

---

## 일상적인 워크플로우 (서버 ↔ 로컬)

### 서버에서 작업 후 로컬로 보내기

```bash
# 서버에서
git add <파일>
git commit -m "메시지"
git push origin main

# 로컬에서
git pull origin main
```

### 로컬에서 작업 후 서버로 보내기

```bash
# 로컬에서
git add <파일>
git commit -m "메시지"
git push origin main

# 서버에서
git pull origin main
```

### 미커밋 변경사항이 있을 때 pull

```bash
git stash
git pull origin main
git stash pop
```

---

## 공식 upstream 업데이트 반영

huggingface/lerobot에 새 기능/수정이 올라왔을 때:

> **주의**: merge 전에 작업 중인 변경사항을 먼저 commit 해두어야 함

```bash
git fetch upstream                 # 공식 최신 정보 가져오기
git merge upstream/main            # 내 main에 merge (충돌 시 수동 해결)
git push origin main               # 내 fork에도 반영
```

### 충돌(conflict) 발생 시

```bash
# 충돌 파일 확인
git status

# 파일 열어서 충돌 수동 해결 후
git add <충돌 해결한 파일>
git merge --continue
```

### 충돌 자주 나는 파일

upstream 업데이트 시 아래 파일에서 충돌 가능성 높음:

- `src/lerobot/policies/__init__.py`
- `src/lerobot/policies/factory.py`
- `examples/training/train_policy.py`

---

## 현재 커스텀 추가 내용

| 경로 | 설명 |
|------|------|
| `src/lerobot/policies/drift/` | Drift policy |
| `src/lerobot/scripts/lerobot_train_robocasa.py` | RoboCasa 학습 스크립트 |
| `src/lerobot/scripts/lerobot_load_robocasa_policy.py` | RoboCasa policy 로드 스크립트 |

---

## Git 계정 정보

```
user.name  = jun981015
user.email = junhyeongg@hanyang.ac.kr
origin     = https://github.com/jun981015/lerobot.git
```

# Lerobot 개발 가이드: Fork + Upstream 방식

이 가이드는 **GitHub Fork**를 활용하여 다음 두 가지를 해결합니다.
1. **서버 ↔ 로컬 동기화**: 내 GitHub(`origin`)를 중계소로 사용하여 IP/방화벽 문제없이 코드 공유.
2. **공식 업데이트 반영**: 공식 레포(`upstream`)의 최신 기능을 내 코드로 안전하게 가져옴.

---

## 1. GitHub 설정 (웹사이트)

1. GitHub의 huggingface/lerobot 페이지로 이동.
2. 우측 상단 **Fork** 버튼 클릭 → 내 계정(`junhyeong`)으로 복사.

---

## 2. 기존 작업 중인 폴더가 있는 경우 (마이그레이션)

이미 작업을 하고 있던 폴더가 있다면, 새로 다운로드받지 말고 **연결만 바꿔주면 됩니다.**

```bash
cd lerobot

# 1. 기존 연결(공식 레포)의 이름을 origin -> upstream으로 변경
git remote rename origin upstream

# 2. 내 포크(junhyeong)를 새로운 origin으로 등록
git remote add origin https://github.com/junhyeong/lerobot.git

# 3. 확인 (origin: 내꺼, upstream: 공식꺼)
git remote -v

# 4. 내 포크로 현재 작업 업로드
git push -u origin main
```

---

**2) 서버 lerobot에 로컬 PC를 remote로 추가 (1회만)**

```bash
cd /home/me/lerobot
git remote add home ssh://LOCAL_USER@LOCAL_HOST/~/repos/lerobot.git
```

---

**3) 서버의 현재 상태를 로컬 PC로 push**

```bash
cd /home/me/lerobot
git push home HEAD
```

---

**4) 로컬 PC에 작업용 클론 만들기 (1회만)**

```bash
cd ~/repos
git clone ~/repos/lerobot.git lerobot-work
cd ~/repos/lerobot-work
```

---

**5) 공식 upstream(원본 lerobot)도 연결**

서버와 로컬 둘 다 같은 upstream을 붙여두면 편합니다.

```bash
git remote add upstream https://github.com/ORG/lerobot.git
```

---

**6) 공식 최신 유지 (서버/로컬 공통)**

```bash
git checkout main
git pull upstream main
```

---

**7) 내 작업은 별도 브랜치로 (권장)**

서버/로컬에서:

```bash
git checkout -b my-work
```

---

**8) 로컬 → 서버로 보내기**

로컬 PC에서:

```bash
cd ~/repos/lerobot-work
git push home my-work
```

---

**9) 서버에서 내 브랜치 받기**

서버에서:

```bash
cd /home/me/lerobot
git fetch home
git checkout my-work
git pull home my-work
```

---

**10) 서버 → 로컬로 보내기**

서버에서:

```bash
git push home my-work
```

---

**11) 로컬에 미커밋 변경이 있을 때 (안전하게 pull)**

로컬에서:

```bash
git stash -u
git pull
git stash pop
```

---

**정리 요약**
- 공식 최신은 `upstream/main`에서 받는다.
- 내 작업은 `my-work` 같은 개인 브랜치에서 한다.
- 서버↔로컬은 bare repo를 통해 push/pull 한다.
- 미커밋 변경이 있으면 `stash`로 안전하게 당겨온다.

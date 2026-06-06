# 🌿 조용한 회복 공간 (A Quiet Recovery Space)

대학생 번아웃 회복을 위한 **RAG 기반 심리 상담 챗봇**

---

## 개요

**Burnout_RAG**는 번아웃을 경험하는 대학생을 위한 한국어 AI 상담 챗봇입니다. 국내 학술 연구 논문 20편을 기반으로 한 RAG(Retrieval-Augmented Generation) 파이프라인을 통해 근거 기반의 회복 정보를 제공하며, 위기 상황에서는 즉각적인 안전 자원을 안내합니다.

### 주요 특징

- **근거 기반 응답**: 번아웃, 스트레스, 회복, 수면 관련 학술 논문을 벡터 DB로 구축하여 신뢰할 수 있는 정보 제공
- **3단계 위기 감지**: 키워드 감지 → 의도 분류 → LLM 검증을 통한 안전 우선 설계
- **하이브리드 검색**: BM25(어휘 검색) + FAISS(의미 검색) 앙상블 + 임베딩 재랭킹
- **Self-RAG 검증**: 생성된 답변이 문서에 의해 지지되는지 자동으로 검증 및 수정
- **마음온도 시각화**: 체크인 설문 기반의 감정 온도(0~100) 추적 및 시각화
- **적응형 상담 톤**: 위험 수준(HIGH/MID/LOW) × 응답 모드(EXPLORE/SUPPORT/GUIDE/INFO)에 따른 맞춤형 응답

---

## 시스템 아키텍처

```
사용자 입력 (채팅)
  │
  ├─ classify_intent()       → INFORMATIONAL / PERSONAL
  ├─ classify_risk()         → HIGH / MID / LOW
  ├─ classify_response_mode() → EXPLORE / SUPPORT / GUIDE / INFO
  │
  ├─ 하이브리드 검색
  │    ├─ BM25Retriever (Kiwi 형태소 분석) → 12개 문서
  │    ├─ FAISS Retriever (벡터 검색)       → 12개 문서
  │    └─ 앙상블 + 임베딩 재랭킹            → 상위 3개
  │
  ├─ LLM 응답 생성 (GPT-4o-mini)
  │    입력: 질문 + 위험 지시문 + 모드 지시문 + 검색 문서 + 대화 이력 + Few-shot
  │
  ├─ Self-RAG 검증
  │    ├─ SupportEval: 문서 지지 수준 평가
  │    ├─ UtilityEval: 응답 유용성 평가 (1~5점)
  │    └─ 검증 실패 → 보수적 재작성
  │
  └─ 최종 응답 구성
       ├─ core_answer    (AI 생성 응답)
       ├─ caveat         (검증 미통과 시 첨부)
       └─ safety_note    (HIGH 위험 + 개인 의도 시 위기 자원 안내)
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **언어** | Python 3.8+ |
| **프론트엔드** | Streamlit |
| **LLM** | OpenAI GPT-4o-mini |
| **임베딩** | OpenAI text-embedding-3-large |
| **벡터 스토어** | FAISS |
| **어휘 검색** | BM25 (rank-bm25) |
| **형태소 분석** | Kiwipiepy (한국어) |
| **RAG 프레임워크** | LangChain |
| **데이터 검증** | Pydantic v2 |

---

## 디렉토리 구조

```
Burnout_RAG/
├── app.py                  # Streamlit 앱 진입점
├── base.py                 # 핵심 RAG 파이프라인 (BaseRAGPipeline, SelfRAGPipeline)
├── prompt_config.py        # 프롬프트 템플릿 및 시스템 지시문
├── requirements.txt        # 의존성 목록
│
├── ui/                     # UI 컴포넌트
│   ├── checkin_insights.py # 체크인 설문 및 마음온도 시각화
│   ├── components.py       # 공통 UI 컴포넌트 (안전 노트 등)
│   ├── layout.py           # 메인 레이아웃 렌더링
│   ├── styles.py           # CSS 커스텀 스타일
│   └── theme.py            # 컬러 팔레트 및 테마 정의
│
├── data/
│   ├── .env                # API 키 설정 (직접 생성 필요)
│   ├── P0_safety/          # 위기 개입 · 자살 예방 논문
│   ├── P1_burnout/         # 번아웃 정의 · 측정 · 회복 논문
│   ├── P2_recovery/        # 회복탄력성 · 마음챙김 논문
│   └── P3_sleep_stress/    # 수면 · 스트레스 논문
│
├── faiss_db/               # FAISS 벡터 인덱스 (자동 생성)
└── .streamlit/
    └── config.toml         # Streamlit 테마 설정
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`data/.env` 파일을 생성하고 OpenAI API 키를 입력합니다.

```env
OPENAI_API_KEY=sk-...
```

### 3. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

> **참고**: 최초 실행 시 FAISS 벡터 인덱스를 자동으로 빌드합니다. 수 분이 소요될 수 있습니다.

---

## 사용 흐름

1. **이름 입력 & 아바타 선택**: 간단한 온보딩
2. **마음 체크인 (6문항)**: 수면, 활력, 회복감, 기분, 스트레스, 피로도
3. **마음온도 확인**: 체크인 결과 기반 감정 온도(0~100) 및 위험 수준 표시
4. **대화 시작**: 번아웃, 스트레스, 회복 등 자유로운 대화
   - AI가 상황에 따라 EXPLORE → SUPPORT → GUIDE → INFO 모드로 적응
   - 위기 신호 감지 시 자동으로 위기 상담 자원 안내

---

## 지식 베이스

| 카테고리 | 우선순위 | 내용 |
|---------|---------|------|
| `P0_safety` | 1 (최우선) | 자살 예방, 위기 개입, 대학생 상담 연구 |
| `P1_burnout` | 2 | 번아웃 정의, MBI 측정, 회복 전략 |
| `P2_recovery` | 3 | 회복탄력성, 마음챙김, 자기돌봄 |
| `P3_sleep_stress` | 4 | 수면의 질, 스트레스 영향, 피로 관리 |

총 **20편**의 한국어 학술 논문으로 구성되어 있습니다.

---

## 위기 지원 자원

이 챗봇은 전문적인 심리 치료를 대체하지 않습니다. 위기 상황 시 즉시 아래 자원에 연락하세요.

| 기관 | 연락처 |
|------|--------|
| 자살예방상담전화 | **109** (24시간) |
| 정신건강위기상담전화 | **1577-0199** (24시간) |

---

## 모델 설정

`base.py` 상단에서 모델을 변경할 수 있습니다.

```python
EMBEDDING_MODEL = "text-embedding-3-large"  # 임베딩 모델
LLM_MODEL       = "gpt-4o-mini"            # 대화 모델
TEMPERATURE     = 0.35                      # 낮은 온도 = 일관된 응답
```

---

## 주의사항

- 이 시스템은 **연구 및 교육 목적**으로 제작되었습니다.
- 의학적 진단이나 전문 심리 치료를 대체하지 않습니다.
- 심각한 위기 상황에서는 반드시 전문가 도움을 받으세요.

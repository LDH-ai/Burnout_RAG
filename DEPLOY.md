# 마음 회복 RAG 챗봇 — 실행 & 배포 가이드

번아웃·수면·스트레스 회복을 돕는 RAG 상담 챗봇. LangChain + FAISS + Streamlit 기반.

---

## 1. 폴더 구조

```
프로젝트/
├── data/
│   ├── .env                  # OPENAI_API_KEY=sk-...  (로컬 전용)
│   ├── P0_safety/            # 위기/안전 논문 PDF
│   ├── P1_burnout/           # 번아웃 PDF
│   ├── P2_recovery/          # 회복 PDF
│   └── P3_sleep_stress/      # 수면·스트레스 PDF
├── faiss_db/                 # 첫 실행 시 자동 생성 (커밋하지 말 것)
├── base.py                   # RAG 파이프라인 (Phase 1~3 + Self-RAG)
├── app.py                    # Streamlit UI
├── ragas_eval.py             # RAGAS 성능 평가
├── requirements.txt          # 앱/배포 의존성
└── requirements-dev.txt      # 평가 전용 의존성
```

`data/.env` 한 줄:
```
OPENAI_API_KEY=sk-여기에_본인_키
```

---

## 2. 로컬 실행

```bash
# 가상환경 권장
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)

pip install -r requirements.txt
streamlit run app.py
```

첫 실행 때 `data/` 의 PDF 를 읽어 FAISS 를 만들고 `faiss_db/` 에 저장한다(시간 소요).
이후 실행은 저장된 인덱스를 로드하므로 빠르다.
문서를 바꿔 재빌드하려면 `faiss_db/` 폴더를 지우고 다시 실행하면 된다.

사이드바에서 검색 엔진(Phase)을 고를 수 있다:
- **Phase 2 · 메모리(권장)** — 기본. 세션별 대화·위험도 누적
- **Phase 1 · 기본 FAISS** — 단일 벡터 검색
- **Phase 3 · Hybrid(BM25+FAISS)** — 앙상블 검색
- **Phase 3 · Hybrid+Self-RAG** — 앙상블 + 자체 검증(환각 방지)

---

## 3. RAGAS 성능 평가

```bash
pip install -r requirements-dev.txt
python ragas_eval.py
```

- `ragas_eval.py` 상단의 `GROUND_TRUTH` 10개를 **본인 적재 문서에 맞게 수정**할 것.
- 결과: 콘솔 표 + `ragas_result.csv` + `ragas_result.png`
- 4대 지표(Faithfulness / Answer Relevancy / Context Precision / Context Recall) 모두 0~1.

낮은 점수 개선 방향(강의 13주):
| 낮은 지표 | 개선 |
|---|---|
| Faithfulness | 프롬프트에 "문맥만 활용" 강조, 환각 방지 지시 |
| Context Precision | 청크 크기 ↓, Top-K 조정, 임베딩 교체 |
| Context Recall | 청크 오버랩 ↑, 검색 K ↑ |
| Answer Relevancy | 시스템 프롬프트·답변 형식 가이드 개선 |

Phase 비교 평가는 `ragas_eval.py` 상단의 `PipelineClass` import 줄만 바꿔 여러 번 실행해 CSV 를 비교한다.

---

## 4. Streamlit Cloud 배포

1. **GitHub(Public) 저장소**에 업로드: `app.py`, `base.py`, `requirements.txt`
   - ⚠️ `data/.env` 와 `faiss_db/` 는 **커밋하지 말 것** (`.gitignore` 에 추가).
   - 배포 환경에서는 `data/` 의 PDF 가 있어야 첫 빌드가 된다.
     PDF 를 저장소에 함께 올리거나, 빌드된 `faiss_db/` 를 올려 빌드를 건너뛸 수 있다.
2. [share.streamlit.io](https://share.streamlit.io) → New app → 저장소·브랜치·`app.py` 지정.
3. **Advanced settings → Secrets** 에 키 입력(TOML 형식):
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
   `app.py` 가 `st.secrets` 의 키를 `os.environ` 으로 넘겨주므로 그대로 동작한다.
4. Deploy.

### `.gitignore` 예시
```
venv/
data/.env
faiss_db/
__pycache__/
*.pyc
ragas_result.*
```

---

## 5. 안전 설계 메모

- 위기 자원 번호는 `base.py` 의 단일 상수(`CRISIS_LINE_SUICIDE=109`, `CRISIS_LINE_MENTAL=1577-0199`)에서 관리된다. 번호가 바뀌면 그 상수만 고치면 앱·답변 양쪽에 반영된다.
- "죽고 싶다/사라지고 싶다" 등 고위험 표현은 체크인 점수가 없어도 텍스트만으로 감지되어 안전 안내로 전환된다.
- Self-RAG Phase 는 답변 근거가 부족하면(환각 의심) 사용자에게 불확실성을 고지한다.
- 본 서비스는 의료 행위가 아니며, 위급 시 전문기관 연계를 안내한다.

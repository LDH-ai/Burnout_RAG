"""
generate_ground_truth.py
========================
실제 적재된 PDF 문서에서 LLM으로 ragas_eval용 Ground Truth Q&A 10개를 생성한다.

실행:
    python generate_ground_truth.py

출력:
    콘솔에 GROUND_TRUTH 리스트 출력 → ragas_eval.py 의 GROUND_TRUTH 에 붙여넣어 사용.
"""

from __future__ import annotations

import json
import os
import random

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from base import Phase1FaissPipeline

load_dotenv("./data/.env")

# 카테고리별 생성 주제 (카테고리 특성에 맞게 질문 방향을 안내)
CATEGORY_TOPICS: dict[str, list[str]] = {
    "safety":       ["자살 위기 신호 인식", "위기 개입 방법", "상담 연계 절차"],
    "burnout":      ["번아웃 정의와 증상", "번아웃 원인과 위험 요인", "완벽주의와 번아웃의 관계"],
    "recovery":     ["번아웃 회복 전략", "자기돌봄 방법", "회복탄력성 강화"],
    "sleep_stress": ["수면과 정서 회복의 관계", "스트레스 대처법", "만성 피로 관리"],
}

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 심리·번아웃 연구 분야 RAG 평가 데이터를 설계하는 전문가입니다.\n"
     "아래 [문서 발췌]에서 직접 답할 수 있는 질문 1개와 모범 정답 1개를 만드세요.\n\n"
     "규칙:\n"
     "- 질문: 실제 사용자가 상담 챗봇에 묻는 자연스러운 한국어 의문문\n"
     "- 정답: 문서에 근거한 완전한 문장, 2~3줄 이내\n"
     "- 의학적 진단·단정 표현 금지, 경향·설명 위주로 작성\n"
     '- 반드시 JSON 형식만 출력: {{"question": "...", "ground_truth": "..."}}'),
    ("human", "[주제 힌트]: {topic}\n\n[문서 발췌]\n{context}"),
])


def _sample_chunks(pipeline: Phase1FaissPipeline, n_per_category: int = 3) -> dict[str, list[str]]:
    """카테고리별로 내용이 충분한 청크를 랜덤 샘플링한다."""
    corpus = pipeline._get_corpus_documents()
    buckets: dict[str, list[str]] = {}
    for doc in corpus:
        cat = doc.metadata.get("category", "burnout")
        text = doc.page_content.strip()
        if len(text) > 200:
            buckets.setdefault(cat, []).append(text)

    result = {}
    for cat, chunks in buckets.items():
        k = min(n_per_category, len(chunks))
        result[cat] = random.sample(chunks, k)
    return result


def generate(n_total: int = 10) -> list[dict[str, str]]:
    """문서에서 LLM으로 ground truth n_total개를 생성한다."""
    print("파이프라인 초기화 중...")
    pipeline = Phase1FaissPipeline()
    pipeline.build_vectorstore()

    llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    chain = GENERATE_PROMPT | llm

    chunks_by_cat = _sample_chunks(pipeline)
    categories    = list(chunks_by_cat.keys())
    per_cat       = max(1, -(-n_total // len(categories)))  # ceiling division

    pairs: list[dict[str, str]] = []

    for cat, chunks in chunks_by_cat.items():
        topics = CATEGORY_TOPICS.get(cat, ["관련 주제"])
        for i, chunk in enumerate(chunks[:per_cat]):
            if len(pairs) >= n_total:
                break
            topic = topics[i % len(topics)]
            print(f"  [{cat}] '{topic}' 생성 중...")
            try:
                response = chain.invoke({"topic": topic, "context": chunk[:1200]})
                data     = json.loads(response.content)
                if "question" in data and "ground_truth" in data:
                    pairs.append({"question": data["question"], "ground_truth": data["ground_truth"]})
            except Exception as exc:
                print(f"    실패: {exc}")

        if len(pairs) >= n_total:
            break

    return pairs[:n_total]


def main() -> None:
    print("=== Ground Truth 자동 생성 시작 ===\n")
    pairs = generate(n_total=10)
    print(f"\n=== 완료: {len(pairs)}개 생성 ===")
    print("아래 코드를 ragas_eval.py 의 GROUND_TRUTH 에 붙여넣으세요.\n")

    print("GROUND_TRUTH: list[dict[str, str]] = [")
    for item in pairs:
        q  = item["question"].replace('"', '\\"').replace("\n", " ")
        gt = item["ground_truth"].replace('"', '\\"').replace("\n", " ")
        print("    {")
        print(f'        "question":     "{q}",')
        print(f'        "ground_truth": "{gt}",')
        print("    },")
    print("]")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        load_dotenv("./data/.env")
    main()

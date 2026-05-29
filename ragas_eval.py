from __future__ import annotations

# ── ragas 0.4.x / langchain_community 0.4.x 호환 패치 ─────────────────────────
# langchain_community 0.4.x에서 vertexai 모듈이 제거됐지만 ragas가 하드코딩으로
# 가져오므로 더미 stub으로 import 오류를 막는다.
import sys
import types as _types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI: pass  # noqa: E301
    _stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
# ──────────────────────────────────────────────────────────────────────────────

import os

from base import Phase3SelfRAGPipeline as PipelineClass
# from base import Phase1FaissPipeline as PipelineClass
# from base import Phase3HybridPipeline as PipelineClass

GROUND_TRUTH: list[dict[str, str]] = [
    {
        "question":     "자살 위기 신호는 어떤 것들이 있나요?",
        "ground_truth": "자살 위기 신호는 다양한 형태로 나타날 수 있으며, 예를 들어 우울한 기분, 사회적 고립, 무기력감, 그리고 일상적인 활동에 대한 흥미 상실 등이 포함될 수 있습니다. 이러한 신호를 인식하는 것이 중요합니다.",
    },
    {
        "question":     "위기 개입 방법에는 어떤 것들이 있나요?",
        "ground_truth": "위기 개입 방법은 다양한 접근 방식을 포함하며, 주로 감정적 지원, 안전한 환경 조성, 그리고 문제 해결을 위한 구체적인 전략을 제공하는 데 중점을 둡니다.",
    },
    {
        "question":     "대학생들이 상담을 받을 때 어떤 문제들이 주로 다뤄지나요?",
        "ground_truth": "대학생들은 면심리적 문제, 이성 문제, 가족 문제, 진로 및 취업 문제, 경제 문제와 관련된 스트레스를 경험하며, 이러한 문제들이 상담에서 주로 다뤄질 수 있습니다.",
    },
    {
        "question":     "번아웃이란 무엇인가요?",
        "ground_truth": "번아웃은 일에서의 지속적인 스트레스와 압박감으로 인해 발생하는 심리적, 정서적 탈진 상태를 의미합니다. 이는 개인이 목표와 성과를 향해 달려가다가 자신의 정신적, 정서적 자원을 소진하게 되는 과정을 포함합니다.",
    },
    {
        "question":     "번아웃의 원인은 무엇인가요?",
        "ground_truth": "번아웃의 원인으로는 직무 자율성 부족, 과도한 경쟁, 엄격한 조직문화, 업무 관행, 권위주의적 업무 환경 등이 있습니다. 또한, 과다한 업무와 정시 퇴근 불가와 같은 업무 특성도 중요한 위험 요인으로 작용할 수 있습니다.",
    },
    {
        "question":     "완벽주의가 번아웃에 어떤 영향을 미칠까요?",
        "ground_truth": "완벽주의는 종종 높은 기대와 압박을 동반하여 스트레스를 증가시킬 수 있으며, 이는 번아웃의 위험 요소로 작용할 수 있습니다. 따라서 완벽주의적인 경향이 있는 사람들은 번아웃을 경험할 가능성이 높아질 수 있습니다.",
    },
    {
        "question":     "번아웃 회복을 위해 어떤 전략이 효과적인가요?",
        "ground_truth": "회복탄력성을 높이기 위해 사회적 지지 체계를 고려한 다양한 프로그램 개발이 필요합니다. 가족, 교수 멘토링, 친구의 지지 등을 활용하여 맞춤형 프로그램을 적용하면 더욱 효과적일 것입니다.",
    },
    {
        "question":     "자기돌봄 방법에는 어떤 것들이 있을까요?",
        "ground_truth": "자기돌봄 방법은 개인의 정서적, 신체적 건강을 증진시키기 위한 다양한 활동을 포함합니다. 예를 들어, 규칙적인 운동, 충분한 수면, 건강한 식습관, 그리고 스트레스를 관리하는 방법들이 있습니다.",
    },
    {
        "question":     "회복탄력성을 높이기 위해 어떤 방법이 있을까요?",
        "ground_truth": "마음챙김 프로그램은 돌봄 종사자들이 스트레스를 완화하고 긍정적인 사고로 전환하는 데 도움을 줄 수 있습니다. 이 프로그램은 자기 돌봄을 촉진하고 균형 있는 삶의 방향을 형성하는 데 기여할 수 있습니다.",
    },
    {
        "question":     "수면이 정서 회복에 어떤 영향을 미치나요?",
        "ground_truth": "수면이 충분하지 않다고 느끼는 경우 피로가 유의하게 높아지는 경향이 있습니다. 이는 정서 회복에 부정적인 영향을 미칠 수 있음을 시사합니다.",
    },
]


def build_eval_dataset(pipeline):
    """각 질문에 대해 검색 문맥·RAG 답변을 수집해 ragas EvaluationDataset을 반환한다."""
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

    samples = []
    for i, item in enumerate(GROUND_TRUTH, start=1):
        q = item["question"]
        print(f"[{i:2d}/{len(GROUND_TRUTH)}] {q}")

        docs   = pipeline.retriever.invoke(q)
        ctx    = [d.page_content for d in docs]
        result = pipeline.ask(q, session_id=f"eval-{i}")

        samples.append(SingleTurnSample(
            user_input=q,
            response=result["answer"],
            retrieved_contexts=ctx,
            reference=item["ground_truth"],
        ))

    return EvaluationDataset(samples=samples)


def main() -> None:
    from ragas import evaluate
    # evaluate()의 isinstance 검사는 ragas.metrics.base.Metric 기반이므로
    # collections API가 아닌 기존 싱글톤 메트릭을 사용해야 한다.
    # llm을 evaluate()에 직접 전달하면 metric.llm is None인 메트릭에 자동 주입된다.
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from langchain_openai import ChatOpenAI

    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    metrics       = [faithfulness, answer_relevancy, context_precision, context_recall]

    print("=== 파이프라인 준비 ===")
    pipeline = PipelineClass()
    pipeline.build_chain()

    print("\n=== 평가 데이터 생성 ===")
    dataset = build_eval_dataset(pipeline)

    print("\n=== RAGAS 평가 실행 ===")
    result  = evaluate(dataset, metrics=metrics, llm=evaluator_llm)

    df   = result.to_pandas()
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n=== 질문별 점수 ===")
    print(df[["user_input"] + cols].to_string(index=False))

    mean_scores = df[cols].mean()
    print("\n=== 평균 점수 ===")
    print(mean_scores.round(3).to_string())

    df.to_csv("ragas_result.csv", index=False, encoding="utf-8-sig")
    print("\n저장: ragas_result.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = mean_scores.plot(
            kind="bar",
            figsize=(8, 4),
            color=["#2563EB", "#0891B2", "#7C3AED", "#D97706"],
        )
        ax.set_title("마음 회복 RAG · RAGAS 평가 결과")
        ax.set_ylim(0, 1.0)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig("ragas_result.png", dpi=150)
        print("저장: ragas_result.png")
    except ImportError:
        print("(matplotlib 미설치 → 그래프 생략)")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv("./data/.env")
    main()

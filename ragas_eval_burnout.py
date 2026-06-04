import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

load_dotenv(".env")
load_dotenv("./data/.env")

from base import MindCareRAGPipeline

# RAGAS import
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")

    questions_df = pd.read_csv("eval_questions.csv")

    rag = MindCareRAGPipeline()
    rag.build_chain()

    rows = []

    for idx, row in questions_df.iterrows():
        question = str(row["question"])
        reference = str(row["reference"])

        print(f"[{idx + 1}/{len(questions_df)}] 평가 데이터 생성 중: {question}")

        # RAG 답변 생성
        result = rag.ask(question, session_id="ragas-eval")
        answer = result["answer"]

        # 검색 context 추출
        docs = rag.retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": reference,
        })

    # 평가 데이터 저장
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv("ragas_eval_dataset.csv", index=False, encoding="utf-8-sig")

    dataset = Dataset.from_list(rows)

    print("\nRAGAS 평가 시작...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    result_df = result.to_pandas()
    result_df.to_csv("ragas_result.csv", index=False, encoding="utf-8-sig")

    summary = result_df[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean().reset_index()

    summary.columns = ["metric", "score"]
    summary.to_csv("ragas_summary.csv", index=False, encoding="utf-8-sig")

    print("\n평가 완료!")
    print(summary)

if __name__ == "__main__":
    main()
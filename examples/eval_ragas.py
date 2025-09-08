import os, json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall

from agent_skeletons.skeletons import build_entrada_transformacao_saida_chain

DATA = [
  {
    "question": "Qual era a meta Selic em setembro de 2025?",
    "contexts": [
      "[doc1] A taxa Selic é definida pelo Copom. Em 2025 variou entre X% e Y%.",
      "[doc3] Em setembro de 2025, a meta Selic estava em Z% (ata do Copom)."
    ],
    "reference": "Em setembro/2025 a meta Selic estava em Z% (conforme ata do Copom)."
  }
]

def run_chain(model: str, question: str):
    os.environ["OLLAMA_MODEL"] = model
    chain = build_entrada_transformacao_saida_chain()
    return chain.invoke({"pergunta": question})

def main():
    models = os.getenv("MODELS", "qwen2.5:3b,llama3.1:8b").split(",")
    results = []
    for m in models:
        answers = []
        for row in DATA:
            out = run_chain(m, row["question"])
            answers.append({
                "question": row["question"],
                "contexts": row["contexts"],
                "answer": out,
                "reference": row["reference"],
            })
        ds = Dataset.from_list(answers)
        scores = evaluate(ds, metrics=[answer_relevancy, faithfulness, context_precision, context_recall])
        avg = {k: float(v) for k, v in scores.items()}
        results.append({"model": m, **avg})
    print("\n=== RAGAS (média por modelo) ===")
    for r in results:
        print(f"{r['model']}: rel={r['answer_relevancy']:.3f} faith={r['faithfulness']:.3f} cprec={r['context_precision']:.3f} crec={r['context_recall']:.3f}")

if __name__ == "__main__":
    main()

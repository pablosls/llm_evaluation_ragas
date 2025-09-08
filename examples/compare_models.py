import os, json, statistics, requests, time, math
from datasets import Dataset

# Try to import RAGAS and its LangChain wrappers; if unavailable, we'll fall back to local metrics
USE_RAGAS = True
try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
    try:
        from ragas.llms import LangchainLLM  # variant A
        from ragas.embeddings import LangchainEmbeddings
    except Exception:
        try:
            from ragas.integrations.langchain import LangchainLLM, LangchainEmbeddings  # variant B
        except Exception:
            USE_RAGAS = False
except Exception:
    USE_RAGAS = False

import numpy as np
from agent_skeletons.skeletons import build_entrada_transformacao_saida_chain, get_llm, get_embeddings

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
MODELS = [m.strip() for m in os.getenv("MODELS", "qwen2.5:3b,llama3.1:8b").split(",") if m.strip()]
PROMPT = os.getenv("PROMPT", "Explique em 4 bullets: como otimizar um job PySpark com shuffle pesado.")
N_RUNS = int(os.getenv("N_RUNS", "3"))
OPTIONS = {"temperature": 0.2, "num_ctx": 4096, "num_predict": 512}

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

def chat_stream(model: str, prompt: str):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True, "options": OPTIONS}
    s = requests.post(url, json=payload, stream=True, timeout=TIMEOUT)
    s.raise_for_status()
    return s.iter_lines()

def time_one(model: str, prompt: str):
    t0 = time.perf_counter()
    ttft, tokens = None, 0
    for raw in chat_stream(model, prompt):
        if not raw: continue
        data = json.loads(raw.decode("utf-8"))
        if "response" in data and data["response"]:
            if ttft is None: ttft = time.perf_counter() - t0
            tokens += len(data["response"].split())
        if data.get("done"): break
    total = time.perf_counter() - t0
    return {"ttft": ttft or total, "total": total, "tokens": tokens, "tps": tokens/total if total>0 else 0}

def bench_model(model: str):
    results = [time_one(model, PROMPT) for _ in range(N_RUNS)]
    def agg(key): xs = [r[key] for r in results]; return statistics.mean(xs)
    return {"ttft": agg("ttft"), "total": agg("total"), "tps": agg("tps"), "tokens_avg": agg("tokens")}

# --------- Local metrics fallback (no RAGAS) ---------
def _cos(a, b):
    a = np.array(a); b = np.array(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def local_quality(answer: str, contexts: list[str], reference: str):
    emb = get_embeddings()
    # Embed strings
    vec_answer = emb.embed_query(answer)
    vec_reference = emb.embed_query(reference)
    vec_contexts = [emb.embed_query(c) for c in contexts]
    # Answer relevancy ~ similarity(answer, reference)
    answer_relevancy_score = max(0.0, _cos(vec_answer, vec_reference))
    # Faithfulness ~ max similarity(answer, any context)
    if vec_contexts:
        faithfulness_score = max(_cos(vec_answer, vc) for vc in vec_contexts)
    else:
        faithfulness_score = 0.0
    # Context precision ~ fraction of contexts that are actually similar to answer
    tau_p = 0.35
    if vec_contexts:
        used = [_cos(vec_answer, vc) >= tau_p for vc in vec_contexts]
        context_precision_score = sum(used) / len(vec_contexts)
    else:
        context_precision_score = 0.0
    # Context recall ~ how well contexts cover the reference (proxy)
    tau_r = 0.35
    if vec_contexts:
        cover = [ _cos(vec_reference, vc) >= tau_r for vc in vec_contexts ]
        context_recall_score = sum(cover) / len(vec_contexts)
    else:
        context_recall_score = 0.0
    # Clamp to [0,1]
    clamp = lambda x: float(max(0.0, min(1.0, x)))
    return {
        "answer_relevancy": clamp(answer_relevancy_score),
        "faithfulness": clamp(faithfulness_score),
        "context_precision": clamp(context_precision_score),
        "context_recall": clamp(context_recall_score),
    }

def eval_quality(model: str):
    os.environ["OLLAMA_MODEL"] = model
    chain = build_entrada_transformacao_saida_chain()
    rows = []
    for row in DATA:
        out = chain.invoke({"pergunta": row["question"]})
        rows.append((out, row["contexts"], row["reference"]))

    if USE_RAGAS:
        # Use RAGAS if available
        from ragas import evaluate
        if 'LangchainLLM' in globals():
            llm = LangchainLLM(get_llm())
        else:
            llm = None  # Fallback to default if available
        if 'LangchainEmbeddings' in globals():
            embeddings_wrapper = LangchainEmbeddings(get_embeddings())
        else:
            embeddings_wrapper = None

        records = [{"question": DATA[i]["question"], "contexts": rows[i][1], "answer": rows[i][0], "reference": rows[i][2]} for i in range(len(rows))]
        ds = Dataset.from_list(records)
        metrics = [answer_relevancy, faithfulness, context_precision, context_recall]
        kwargs = {}
        if llm is not None: kwargs["llm"] = llm
        if embeddings_wrapper is not None: kwargs["embeddings"] = embeddings_wrapper
        scores = evaluate(ds, metrics=metrics, **kwargs)
        return {k: float(v) for k, v in scores.items()}
    else:
        # Local heuristic metrics
        # Aggregate mean across all rows
        scores = {"answer_relevancy": [], "faithfulness": [], "context_precision": [], "context_recall": []}
        for ans, ctxs, ref in rows:
            s = local_quality(ans, ctxs, ref)
            for k, v in s.items(): scores[k].append(v)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}

def main():
    table = []
    for m in MODELS:
        row = {"model": m}
        try: row.update(bench_model(m))
        except Exception as e: row["bench_error"] = str(e)
        try: row.update(eval_quality(m))
        except Exception as e: row["quality_error"] = str(e)
        table.append(row)

    print("\n=== Comparação Latência vs Qualidade ===")
    header = ["Model","Total(s)","TPS","Faith","Rel","CPrec","CRec"]
    print("{:<15} {:<8} {:<6} {:<6} {:<6} {:<6} {:<6}".format(*header))
    for r in table:
        total = r.get("total", float("nan"))
        tps = r.get("tps", float("nan"))
        f = r.get("faithfulness", float("nan"))
        rel = r.get("answer_relevancy", float("nan"))
        cprec = r.get("context_precision", float("nan"))
        crec = r.get("context_recall", float("nan"))
        print("{:<15} {:<8.2f} {:<6.1f} {:<6.2f} {:<6.2f} {:<6.2f} {:<6.2f}".format(r["model"], total, tps, f, rel, cprec, crec))

    print("\nJSON bruto:")
    print(json.dumps(table, indent=2))
    
    import csv
    with open("compare_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for row in table for k in row.keys()}))
        w.writeheader(); w.writerows(table)
    print("Salvo em compare_results.csv")
    

if __name__ == "__main__":
    main()

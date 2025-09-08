import os
from agent_skeletons.skeletons import (
    build_problema_passos_resposta_chain,
    build_contexto_hipoteses_chain,
    build_pros_contras_decisao_chain,
    build_entrada_transformacao_saida_chain,
    build_ppr_graph,
    build_chec_graph,
    build_pcd_graph,
    build_ets_graph,
)

def main():
    retr = os.getenv("RETRIEVER_KIND", "dummy")
    print(f"[examples] RETRIEVER_KIND={retr}")

    print("=== Runnable: PPR ===")
    ppr = build_problema_passos_resposta_chain()
    print(ppr.invoke({
        "dominio":"engenharia de dados",
        "problema":"Otimizar job PySpark",
        "requisitos":"Spark 3.5",
    }))

    print("\n=== Runnable: CHEC ===")
    chec = build_contexto_hipoteses_chain()
    print(chec.invoke({"contexto":"Falha no treinamento SageMaker"}))

    print("\n=== Runnable: PCD ===")
    pcd = build_pros_contras_decisao_chain()
    print(pcd.invoke({
        "cenario":"Vector DB para POC vs produção",
        "opcoes":["Weaviate","Pinecone"],
        "criterios":["latencia","custo","filtros"],
    }))

    print("\n=== Runnable: ETS (mini-RAG) ===")
    ets = build_entrada_transformacao_saida_chain()
    print(ets.invoke({"pergunta":"Qual a Selic em setembro de 2025?"}))

    print("\n=== Graph: PPR ===")
    g1 = build_ppr_graph()
    print(g1.invoke({
        "dominio":"engenharia de dados",
        "problema":"Otimizar PySpark",
        "requisitos":"custo baixo",
    }))

    print("\n=== Graph: ETS ===")
    g2 = build_ets_graph()
    print(g2.invoke({"pergunta":"Qual a Selic em setembro de 2025?"}))

if __name__ == "__main__":
    main()

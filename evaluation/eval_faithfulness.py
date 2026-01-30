import json
from rag.rag_pipeline import run_rag
from index.search import FaissRetriever

def normalize(s: str) -> str:
    return " ".join(s.lower().split())

def main():
    retriever = FaissRetriever(
        index_path="index/chunk_index.faiss",
        meta_path="index/chunk_metadata.json",
    )

    gold = json.load(open("evaluation/gold_rag_eval.json", "r", encoding="utf-8"))

    faith_ok = 0
    total = 0

    for g in gold:
        query = g["query"]

        # Retrieve the exact context we expect the model to use
        chunks = retriever.retrieve(query, top_k=5)

        # Build one big context string for substring checks
        context = "\n\n".join([f"[{c['doc_id']}|chunk={c['chunk_id']}]\n{c['text']}" for c in chunks])
        context_norm = normalize(context)

        out = run_rag(query, top_k=5)
        total += 1

        # If refusal, consider faithful (it didn't invent)
        if out.answer.strip().lower() == "i don't know":
            faith_ok += 1
            print("\nQuery:", query)
            print("✅ Faithful (refused)")
            continue

        # Evidence checks:
        # 1) must include at least 1 quote
        # 2) every quote must be a substring of context
        evidence_ok = True

        if not getattr(out, "evidence", None) or len(out.evidence) == 0:
            evidence_ok = False
        else:
            for quote in out.evidence:
                if normalize(quote) not in context_norm:
                    evidence_ok = False
                    break

        faith_ok += int(evidence_ok)

        print("\nQuery:", query)
        print("Answer:", out.answer)
        print("Evidence:", getattr(out, "evidence", []))
        print("Faithfulness:", "✅" if evidence_ok else "❌")

        if not evidence_ok:
            print("---- Debug hint ----")
            print("Model either did not quote OR quoted text not present in retrieved context.")

    print("\n=== SUMMARY ===")
    print(f"Faithfulness (evidence-based): {faith_ok}/{total} = {faith_ok/total:.2%}")

if __name__ == "__main__":
    main()

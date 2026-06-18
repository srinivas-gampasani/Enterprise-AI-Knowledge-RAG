#!/usr/bin/env python3
"""
Ascension Via Christi RAG System — Evaluation Script

Runs a benchmark of predefined Q&A pairs against the RAG system
and reports accuracy, latency, and citation quality.

Usage:
    cd backend && python ../scripts/evaluate_rag.py
"""
import asyncio
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

from app.services.rag_service import RAGService
from app.core.config import settings

# Ground-truth evaluation set
EVAL_SET = [
    {
        "question": "What is the sepsis 3-hour bundle?",
        "expected_keywords": ["lactate", "blood cultures", "antibiotics", "crystalloid", "30 mL/kg"],
        "expected_source": "clinical_guidelines.txt",
    },
    {
        "question": "What are the STEMI door-to-balloon time targets?",
        "expected_keywords": ["90 minutes", "Cath Lab", "Aspirin", "Ticagrelor"],
        "expected_source": "clinical_guidelines.txt",
    },
    {
        "question": "What is the heparin infusion therapeutic aPTT range?",
        "expected_keywords": ["50", "70", "therapeutic", "aPTT"],
        "expected_source": "drug_formulary.txt",
    },
    {
        "question": "What medications require a double-check by two nurses?",
        "expected_keywords": ["insulin", "heparin", "opioid", "high-alert"],
        "expected_source": "drug_formulary.txt",
    },
    {
        "question": "How do I report a HIPAA breach?",
        "expected_keywords": ["Privacy Officer", "60 days", "HHS", "OCR", "1 hour"],
        "expected_source": "hr_operational_policies.txt",
    },
    {
        "question": "What are PTO accrual rates for 6-10 years of service?",
        "expected_keywords": ["6.46", "168", "21 days"],
        "expected_source": "hr_operational_policies.txt",
    },
    {
        "question": "What is the EPIC downtime procedure?",
        "expected_keywords": ["binder", "scan", "4 hours", "restoration"],
        "expected_source": "it_ehr_reference.txt",
    },
    {
        "question": "What telehealth platforms are approved at Ascension?",
        "expected_keywords": ["Amwell", "Zoom for Healthcare", "HIPAA BAA"],
        "expected_source": "it_ehr_reference.txt",
    },
    {
        "question": "What is the Morse Fall Scale high risk threshold?",
        "expected_keywords": ["45", "bed alarm", "hourly", "yellow"],
        "expected_source": "clinical_guidelines.txt",
    },
    {
        "question": "What is the warfarin target INR for atrial fibrillation?",
        "expected_keywords": ["2.0", "3.0", "atrial fibrillation"],
        "expected_source": "drug_formulary.txt",
    },
]


def score_answer(answer: str, keywords: list) -> float:
    """Keyword hit rate as a proxy for accuracy."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def check_citation(citations: list, expected_source: str) -> bool:
    """Check if expected source appears in top citations."""
    return any(c['source_file'] == expected_source for c in citations)


async def run_eval():
    print("=" * 65)
    print("  Ascension Via Christi — RAG Evaluation Suite")
    print("=" * 65)

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith('sk-your'):
        print("ERROR: OPENAI_API_KEY not set in backend/.env")
        sys.exit(1)

    print("Initializing RAG service...")
    rag = RAGService()
    await rag.initialize()

    if not rag.is_ready:
        print("ERROR: RAG not ready. Run ingest_documents.py first.")
        sys.exit(1)

    stats = rag.get_stats()
    print(f"Ready — {stats['total_chunks_indexed']} chunks indexed\n")

    results = []
    latencies = []

    for i, item in enumerate(EVAL_SET, 1):
        print(f"[{i:2d}/{len(EVAL_SET)}] {item['question'][:60]}...")
        t0 = time.time()
        response = await rag.query(item['question'], top_k=5)
        latency = (time.time() - t0) * 1000

        kw_score = score_answer(response.answer, item['expected_keywords'])
        cited_correctly = check_citation(response.citations, item['expected_source'])
        latencies.append(latency)

        result = {
            "question": item['question'],
            "keyword_score": round(kw_score, 3),
            "citation_correct": cited_correctly,
            "latency_ms": round(latency, 1),
            "tokens": response.tokens_used,
            "chunks_retrieved": response.chunks_retrieved,
        }
        results.append(result)

        status = "PASS" if kw_score >= 0.6 and cited_correctly else "PARTIAL" if kw_score >= 0.4 else "FAIL"
        print(f"       KW Score: {kw_score:.0%} | Citation: {'OK' if cited_correctly else 'MISS'} | Latency: {latency:.0f}ms | [{status}]")

    # Summary
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)

    avg_kw    = sum(r['keyword_score'] for r in results) / len(results)
    pct_cited = sum(1 for r in results if r['citation_correct']) / len(results)
    avg_lat   = sum(latencies) / len(latencies)
    p95_lat   = sorted(latencies)[int(len(latencies)*0.95)]
    passes    = sum(1 for r in results if r['keyword_score'] >= 0.6 and r['citation_correct'])

    print(f"  Questions evaluated : {len(EVAL_SET)}")
    print(f"  Pass rate           : {passes}/{len(EVAL_SET)} ({passes/len(EVAL_SET):.0%})")
    print(f"  Avg keyword score   : {avg_kw:.1%}")
    print(f"  Citation accuracy   : {pct_cited:.1%}")
    print(f"  Avg latency         : {avg_lat:.0f}ms")
    print(f"  p95 latency         : {p95_lat:.0f}ms")
    print(f"  Avg tokens/query    : {sum(r['tokens'] for r in results) // len(results)}")
    print("=" * 65)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'eval_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            "summary": {
                "total": len(EVAL_SET),
                "passes": passes,
                "avg_keyword_score": round(avg_kw, 4),
                "citation_accuracy": round(pct_cited, 4),
                "avg_latency_ms": round(avg_lat, 1),
                "p95_latency_ms": round(p95_lat, 1),
            },
            "results": results
        }, f, indent=2)
    print(f"\n  Results saved to: docs/eval_results.json")


if __name__ == '__main__':
    asyncio.run(run_eval())

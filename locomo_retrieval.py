"""LoCoMo evidence retrieval probe for a locally downloaded official dataset.

IMPORTANT: This is NOT the original FadeMem algorithm, nor LoCoMo answer accuracy.
It is a low-cost, no-LLM preliminary retrieval experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics
import time
from cmb import tokens


def load_locomo(path: str):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict) and isinstance(data.get('samples'), list):
        data = data['samples']
    if not isinstance(data, list):
        raise ValueError('Expected LoCoMo JSON list or {"samples": [...]}')
    return data


def messages(sample):
    conv = sample['conversation']
    sessions = sorted((k for k in conv if re.fullmatch(r'session_\d+', k)),
                      key=lambda k: int(k.split('_')[1]))
    msg = []
    for sess in sessions:
        session_no = int(sess.split('_')[1])
        for turn in conv[sess]:
            if not isinstance(turn, dict) or not turn.get('dia_id'):
                continue
            t = f"{turn.get('speaker','')}: {turn.get('text','')}"
            if turn.get('blip_caption'):
                t += f" Image caption: {turn['blip_caption']}"
            msg.append({'id': turn['dia_id'], 'text': t, 'session': session_no,
                        'bag': Counter(tokens(t))})
    return msg


def keep_proxy(mem, budget):
    """Post-hoc salience proxy (not authentic FadeMem; no access to QA gold labels)."""
    max_session = max((m['session'] for m in mem), default=0)
    scores = []
    observed = Counter()
    for m in mem:
        text = m['text']
        terms = m['bag']
        # Imperfect surrogates for informativeness, resurfacing and recency.
        specific = min(1., sum(ch.isdigit() for ch in text) / 3 +
                       sum(bool(w[:1].isupper()) for w in text.split()) / 12)
        informative = min(1., len(terms) / 22)
        recurrence = min(1., sum(1 for w in terms if observed[w] > 0) / max(1, len(terms)))
        importance = .45 * informative + .30 * specific + .25 * recurrence
        recency = math.exp(-.055 * (max_session - m['session']))
        scores.append(.72 * importance + .28 * recency)
        observed.update(terms)
    winners = sorted(range(len(mem)), key=lambda i: (-scores[i], i))[:budget]
    return [mem[i] for i in sorted(winners)]


def bm25(query, docs, k):
    if not docs:
        return []
    qt = tokens(query)
    dfs = Counter()
    for d in docs:
        dfs.update(d['bag'].keys())
    n = len(docs)
    avgdl = sum(sum(d['bag'].values()) for d in docs) / n
    ranked = []
    for d in docs:
        dl = sum(d['bag'].values())
        score = 0.
        for term in qt:
            tf = d['bag'].get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - dfs[term] + .5) / (dfs[term] + .5))
            score += idf * (tf * 2.2) / (tf + 1.2 * (.25 + .75 * dl / max(1, avgdl)))
        if score > 0:
            ranked.append((score, d['id']))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [docid for _, docid in ranked[:k]]


def evidence_ids(qa):
    out = []
    for element in qa.get('evidence', []):
        if isinstance(element, str):
            out.extend(re.findall(r'D\d+:\d+', element))
    return set(out)


def run(dataset, k=10, retain=.55, max_samples=10):
    details = []
    samples = dataset[:max_samples]
    for index, sample in enumerate(samples):
        corpus = messages(sample)
        budget = max(1, round(len(corpus) * retain))
        strategies = {
            'StoreAll': corpus,
            'FIFO': corpus[-budget:],
            'SalienceProxy_NOT_FadeMem': keep_proxy(corpus, budget),
        }
        corpus_ids = {m['id'] for m in corpus}
        valid_qas = []
        for q in sample.get('qa', []):
            if q.get('category') not in (1, 2, 3, 4):
                continue
            evidence = evidence_ids(q)
            if not evidence or not evidence.issubset(corpus_ids):
                continue
            valid_qas.append((q, evidence))
        for name, docs in strategies.items():
            start = time.perf_counter()
            for qi, (q, evidence) in enumerate(valid_qas):
                found = bm25(q['question'], docs, k)
                recalled = len(set(found) & evidence)
                details.append({
                    'sample_id': str(sample.get('sample_id', index)),
                    'question_index': qi, 'category': q['category'], 'strategy': name,
                    'evidence_recall': recalled / len(evidence),
                    'all_evidence_hit': float(recalled == len(evidence)),
                    'first_evidence_mrr': next((1.0 / (j+1) for j, doc_id in enumerate(found)
                                                if doc_id in evidence), 0.),
                    'messages_total': len(corpus), 'messages_retained': len(docs),
                    'retained_fraction': len(docs) / max(1, len(corpus)),
                })
            _elapsed_ms = (time.perf_counter() - start) * 1000
            print(f'Sample {index+1}/{len(samples)} {name}: {len(valid_qas)} questions in {_elapsed_ms:.0f} ms')
    return details


def main():
    p = argparse.ArgumentParser(description='LoCoMo evidence-retrieval proxy benchmark, NOT an official FadeMem reproduction')
    p.add_argument('--dataset', required=True, help='path to locomo10.json from snap-research/LoCoMo')
    p.add_argument('--output', default='results/locomo_evidence')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--retention', type=float, default=.55)
    p.add_argument('--samples', type=int, default=10)
    args = p.parse_args()
    if not (0 < args.retention <= 1) or min(args.k, args.samples) < 1:
        p.error('0 < retention <= 1 and k,samples >= 1')
    results = run(load_locomo(args.dataset), args.k, args.retention, args.samples)
    if not results:
        raise RuntimeError('No scorable questions with evidence found')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, results[0].keys()); w.writeheader(); w.writerows(results)
    grouped = {}
    for name in sorted(set(r['strategy'] for r in results)):
        r = [row for row in results if row['strategy'] == name]
        grouped[name] = {k: round(statistics.mean(row[k] for row in r), 4) for k in
                         ('evidence_recall', 'all_evidence_hit', 'first_evidence_mrr', 'retained_fraction')}
    target.with_suffix('.json').write_text(json.dumps({'config': vars(args), 'results': grouped}, indent=2))
    print(json.dumps(grouped, indent=2))


if __name__ == '__main__':
    main()

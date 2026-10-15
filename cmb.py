"""Cognitive Memory Benchmark: independent, deterministic FadeMem-inspired prototype.

This is NOT the authors' official FadeMem implementation nor reproduction of their scores.
Python 3.10+; no third-party packages or API keys required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ATTRS = {
    'linguagem': ['Python', 'Rust', 'Go', 'Julia', 'C++'],
    'editor': ['Neovim', 'VS Code', 'Zed', 'Emacs', 'Helix'],
    'musica': ['jazz', 'samba', 'rock', 'bossa nova', 'soul'],
    'instrumento': ['guitarra', 'saxofone', 'piano', 'baixo', 'bateria'],
    'esporte': ['jiu-jitsu', 'futebol', 'xadrez', 'corrida', 'ciclismo'],
    'bebida': ['cha', 'cafe', 'agua', 'suco', 'mate'],
}


def tokens(s: str) -> set[str]:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return set(re.findall(r'[a-z0-9_]+', s))


@dataclass(frozen=True)
class Event:
    id: str
    entity: str
    attr: str
    value: str
    day: int
    critical: bool
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    entity: str
    attrs: tuple[str, ...]
    answers: tuple[str, ...]
    kind: str  # current, historical, multihop
    text: str


def make_dataset(profiles: int, seed: int) -> tuple[list[Event], list[Question]]:
    rng = random.Random(seed)
    es: list[Event] = []
    qs: list[Question] = []
    for i in range(profiles):
        entity = f'Pessoa_{i:03d}'
        initial: dict[str, str] = {}
        newest: dict[str, str] = {}
        for attr, values in ATTRS.items():
            value = rng.choice(values)
            initial[attr] = newest[attr] = value
            day = rng.randint(0, 4)
            es.append(Event(f'{entity}-{attr}-first', entity, attr, value, day, True,
                            f'{entity} informou sua preferencia de {attr}: {value}.'))
        # Repeated statements: a real memory manager should consolidate duplicates.
        for attr in ('musica', 'esporte', 'bebida'):
            value = initial[attr]
            for day in (12, 21):
                es.append(Event(f'{entity}-{attr}-repeat-{day}', entity, attr, value, day,
                                True, f'{entity} confirmou a preferencia de {attr}: {value}.'))
        # Evolving facts: historical statements should not be confused with current state.
        for attr in ('linguagem', 'editor'):
            options = [v for v in ATTRS[attr] if v != initial[attr]]
            value = rng.choice(options)
            newest[attr] = value
            day = rng.randint(22, 26)
            es.append(Event(f'{entity}-{attr}-update', entity, attr, value, day, True,
                            f'{entity} atualizou sua preferencia de {attr} para {value}.'))
        # Low-signal, one-off anecdotes; never used as benchmark question answers.
        for j in range(12):
            day = rng.randint(1, 26)
            attr = f'casual_{j}'
            es.append(Event(f'{entity}-noise-{j}', entity, attr, f'episodio_{j}_{i}',
                            day, False, f'{entity} comentou brevemente sobre {attr} episodio_{j}_{i}.'))
        for attr in ATTRS:
            qs.append(Question(f'{entity}-current-{attr}', entity, (attr,), (newest[attr],),
                               'current', f'Qual e a preferencia atual de {attr} da {entity}?'))
        for attr in ('linguagem', 'editor'):
            qs.append(Question(f'{entity}-historical-{attr}', entity, (attr,), (initial[attr],),
                               'historical', f'Qual era a preferencia original de {attr} da {entity}?'))
        qs.append(Question(f'{entity}-multihop', entity, ('musica', 'instrumento'),
                           (newest['musica'], newest['instrumento']), 'multihop',
                           f'Quais sao as preferencias de musica e instrumento da {entity}?'))
    es.sort(key=lambda x: (x.day, x.id))
    return es, qs


@dataclass
class Memory:
    id: str
    entity: str
    attr: str
    value: str
    text: str
    first_day: int
    last_day: int
    critical: bool
    strength: float = 1.0
    base_importance: float = 0.8
    accesses: int = 0
    layer: str = 'SML'
    last_decay_day: int = 0
    sources: list[str] = field(default_factory=list)
    superseded: bool = False
    terms: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def from_event(cls, ev: Event) -> 'Memory':
        m = cls(ev.id, ev.entity, ev.attr, ev.value, ev.text, ev.day, ev.day,
                ev.critical, 0.98 if ev.critical else 0.58,
                0.8 if ev.critical else 0.12, 0,
                'LML' if ev.critical else 'SML', ev.day, [ev.id])
        m.terms = tokens(m.text)
        return m

    def importance(self, day: int) -> float:
        recent = math.exp(-0.12 * max(0, day - self.last_day))
        return min(1, .82 * self.base_importance + .10 * (self.accesses / (2 + self.accesses)) + .08 * recent)

    def decay(self, day: int, *, layer_enabled: bool = True) -> None:
        elapsed = max(0, day - self.last_decay_day)
        if elapsed == 0:
            return
        importance = self.importance(day)
        if layer_enabled:
            if importance >= .7:
                self.layer = 'LML'
            elif importance < .3:
                self.layer = 'SML'
        else:
            self.layer = 'SML'
        rate = .10 * math.exp(-1.5 * importance)
        beta = 0.8 if self.layer == 'LML' else 1.2
        self.strength *= math.exp(-rate * elapsed ** beta)
        self.last_decay_day = day

    def size_bytes(self) -> int:
        # Actual serialized UTF-8 payload, excluding python object overhead.
        payload = {k: getattr(self, k) for k in
                   ('id', 'entity', 'attr', 'value', 'text', 'first_day', 'last_day',
                    'critical', 'strength', 'base_importance', 'accesses',
                    'layer', 'sources', 'superseded')}
        return len(json.dumps(payload, ensure_ascii=False).encode('utf8'))


class Engine:
    def __init__(self, name: str, *, fifo_limit: int | None = None,
                 decay: bool = False, fusion: bool = False,
                 layer_enabled: bool = True):
        self.name = name
        self.fifo_limit = fifo_limit
        self.decay_enabled = decay
        self.fusion_enabled = fusion
        self.layer_enabled = layer_enabled
        self.mem: list[Memory] = []
        self.pruned = 0
        self.fused = 0

    def advance(self, day: int) -> None:
        if not self.decay_enabled:
            return
        for m in self.mem:
            m.decay(day, layer_enabled=self.layer_enabled)
        before = len(self.mem)
        self.mem = [m for m in self.mem if m.strength >= 0.10]
        self.pruned += before - len(self.mem)

    def ingest(self, ev: Event) -> None:
        m = Memory.from_event(ev)
        # Superseded is metadata, not destructive deletion. Needed for historical tests.
        for old in self.mem:
            if old.entity == ev.entity and old.attr == ev.attr and old.value != ev.value:
                old.superseded = True
        if self.fusion_enabled:
            for old in reversed(self.mem):
                if old.entity == ev.entity and old.attr == ev.attr and old.value == ev.value and not old.superseded:
                    old.sources.append(ev.id)
                    old.last_day = ev.day
                    old.accesses += 1
                    old.strength = min(1., old.strength + .20 * (1 - old.strength))
                    old.text += f' Confirmado no dia {ev.day}.'
                    old.terms = tokens(old.text)
                    self.fused += 1
                    return
        self.mem.append(m)
        if self.fifo_limit and len(self.mem) > self.fifo_limit:
            self.mem = self.mem[-self.fifo_limit:]

    def search(self, q: Question, k: int) -> list[Memory]:
        qt = tokens(q.text)
        results = []
        for m in self.mem:
            # Shared retrieval scoring function for every memory strategy.
            shared = len(qt & m.terms)
            if not shared:
                continue
            score = shared / (math.sqrt(len(qt) * len(m.terms)) or 1)
            if m.entity == q.entity:
                score += 1.5
            if m.attr in q.attrs:
                score += 1.0
            if q.kind == 'historical':
                score += .3 * (30 - m.first_day) / 30
            else:
                score += .3 * m.last_day / 30
            results.append((score, m.last_day, m))
        results.sort(key=lambda row: (row[0], -row[1] if q.kind == 'historical' else row[1]), reverse=True)
        return [m for _, _, m in results[:k]]


def evaluate_hit(q: Question, got: list[Memory]) -> dict[str, float]:
    predicted: list[str] = []
    for attr in q.attrs:
        options = [m for m in got if m.entity == q.entity and m.attr == attr]
        if not options:
            predicted.append('')
        else:
            best = min(options, key=lambda m: m.first_day) if q.kind == 'historical' else max(options, key=lambda m: m.last_day)
            predicted.append(best.value)
    correct = float(tuple(predicted) == q.answers)
    relevant = sum(m.entity == q.entity and m.attr in q.attrs and m.value in q.answers for m in got)
    return {'hit': correct, 'relevance_precision': relevant / len(got) if got else 0.0}


def execute(profiles: int, seed: int, *, k: int = 8, fifo: int = 250):
    events, questions = make_dataset(profiles, seed)
    strategies = [
        Engine('StoreAll'), Engine('FIFO', fifo_limit=fifo),
        Engine('FadeMemInspired', decay=True, fusion=True),
        Engine('FadeNoFusion', decay=True, fusion=False),
        Engine('FadeNoForgetting', fusion=True),
    ]
    day = 0
    ingestion_timings: dict[str, float] = {}
    for strategy in strategies:
        start = time.perf_counter()
        day = 0
        for event in events:
            if event.day != day:
                strategy.advance(event.day)
                day = event.day
            strategy.ingest(event)
        strategy.advance(30)
        ingestion_timings[strategy.name] = (time.perf_counter() - start) * 1000

    rows = []
    summaries = []
    for strategy in strategies:
        latency = []
        for q in questions:
            t0 = time.perf_counter_ns()
            got = strategy.search(q, k)
            latency.append((time.perf_counter_ns() - t0) / 1e6)
            metric = evaluate_hit(q, got)
            rows.append({'seed': seed, 'strategy': strategy.name, 'question': q.id,
                         'kind': q.kind, **metric})
        grouped = defaultdict(list)
        for r in rows:
            if r['seed'] == seed and r['strategy'] == strategy.name:
                grouped[r['kind']].append(r['hit'])
        summaries.append({
            'seed': seed, 'strategy': strategy.name, 'events': len(events),
            'retained': len(strategy.mem), 'storage_reduction_pct':
                100 * (1 - len(strategy.mem) / len(events)),
            'retained_bytes': sum(m.size_bytes() for m in strategy.mem),
            'fused': strategy.fused, 'pruned': strategy.pruned,
            'current_accuracy_pct': 100 * statistics.mean(grouped['current']),
            'historical_accuracy_pct': 100 * statistics.mean(grouped['historical']),
            'multihop_accuracy_pct': 100 * statistics.mean(grouped['multihop']),
            'precision_at_k_pct': 100 * statistics.mean(r['relevance_precision'] for r in rows
                if r['seed'] == seed and r['strategy'] == strategy.name),
            'mean_retrieval_ms': statistics.mean(latency),
            'p95_retrieval_ms': sorted(latency)[int(.95*(len(latency)-1))],
            'ingestion_ms': ingestion_timings[strategy.name],
        })
    return rows, summaries


def write_csv(path: Path, records: list[dict]):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)


def main():
    ap = argparse.ArgumentParser(description='Independent FadeMem-inspired benchmark; no API key needed')
    ap.add_argument('--profiles', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--fifo', type=int, default=250)
    ap.add_argument('--output', default='results')
    args = ap.parse_args()
    if min(args.profiles, args.seeds, args.k, args.fifo) < 1:
        ap.error('profiles, seeds, k and fifo must be >= 1')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    details, summaries = [], []
    for seed in range(args.seeds):
        d, s = execute(args.profiles, seed, k=args.k, fifo=args.fifo)
        details.extend(d)
        summaries.extend(s)
    write_csv(out / 'per_question.csv', details)
    write_csv(out / 'per_seed.csv', summaries)
    aggregated = []
    for name in dict.fromkeys(row['strategy'] for row in summaries):
        group = [r for r in summaries if r['strategy'] == name]
        agg = {'strategy': name}
        for metric in ('storage_reduction_pct', 'current_accuracy_pct', 'historical_accuracy_pct',
                       'multihop_accuracy_pct', 'precision_at_k_pct', 'mean_retrieval_ms',
                       'p95_retrieval_ms', 'ingestion_ms', 'retained_bytes'):
            agg[metric] = round(statistics.mean(r[metric] for r in group), 3)
        aggregated.append(agg)
    write_csv(out / 'summary.csv', aggregated)
    with (out / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump({'profiles': args.profiles, 'seeds': args.seeds, 'k': args.k,
                   'fifo': args.fifo, 'end_day': 30,
                   'methodology': 'synthetic, deterministic, non-official, no LLM'}, f, indent=2)
    print(f'Benchmark finalizado: {len(details)} avaliacoes em {out}/')
    print(f'{"Estrategia":<20} {"Reduz.%":>8} {"Atual%":>8} {"Hist%":>8} {"Multi%":>8} {"Prec.%":>8}')
    for r in aggregated:
        print(f'{r["strategy"]:<20} {r["storage_reduction_pct"]:>8.1f} {r["current_accuracy_pct"]:>8.1f} '
              f'{r["historical_accuracy_pct"]:>8.1f} {r["multihop_accuracy_pct"]:>8.1f} '
              f'{r["precision_at_k_pct"]:>8.1f}')


if __name__ == '__main__':
    main()

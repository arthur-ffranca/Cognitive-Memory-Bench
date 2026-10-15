import unittest
from cmb import Engine, Event, Memory, Question, evaluate_hit, execute, make_dataset


class TestCMB(unittest.TestCase):
    def test_dataset_is_reproducible(self):
        self.assertEqual(make_dataset(3, 7), make_dataset(3, 7))

    def test_dataset_has_updates_and_questions(self):
        events, questions = make_dataset(3, 7)
        self.assertEqual(sum(q.kind == 'historical' for q in questions), 6)
        self.assertEqual(sum(e.attr == 'linguagem' and e.entity == 'Pessoa_000' for e in events), 2)

    def test_merges_identical_information(self):
        e = Engine('fade', fusion=True)
        for d in (1, 7):
            e.ingest(Event(str(d), 'S', 'instrumento', 'piano', d, True, 'S prefere instrumento piano'))
        self.assertEqual(len(e.mem), 1)
        self.assertEqual(len(e.mem[0].sources), 2)

    def test_preserves_temporal_versions(self):
        e = Engine('fade', fusion=True)
        e.ingest(Event('a', 'S', 'linguagem', 'Python', 1, True, 'S prefere linguagem Python'))
        e.ingest(Event('b', 'S', 'linguagem', 'Rust', 20, True, 'S prefere linguagem Rust'))
        current = Question('q', 'S', ('linguagem',), ('Rust',), 'current', 'S linguagem atual')
        original = Question('p', 'S', ('linguagem',), ('Python',), 'historical', 'S linguagem original')
        self.assertEqual(evaluate_hit(current, e.search(current, 8))['hit'], 1.0)
        self.assertEqual(evaluate_hit(original, e.search(original, 8))['hit'], 1.0)

    def test_decay_reduces_noise(self):
        e = Engine('fade', decay=True, fusion=True)
        for d in range(4):
            e.ingest(Event(str(d), 'S', f'casual_{d}', str(d), 0, False, f'S comentou casual_{d}'))
        e.advance(30)
        self.assertLess(len(e.mem), 4)

    def test_all_strategies_run(self):
        records, summaries = execute(2, 0, k=6, fifo=12)
        self.assertEqual(len(summaries), 5)
        self.assertTrue(records)
        self.assertEqual(next(r for r in summaries if r['strategy'] == 'StoreAll')['storage_reduction_pct'], 0)


if __name__ == '__main__':
    unittest.main()

class TestLoCoMoAdapter(unittest.TestCase):
    def test_locomo_adapter_evidence(self):
        from locomo_retrieval import messages, evidence_ids, run
        fixture = [{'sample_id': 'toy', 'conversation': {
            'speaker_a': 'A', 'speaker_b': 'B', 'session_1': [
                {'speaker': 'A', 'dia_id': 'D1:1', 'text': 'I like jazz music'},
                {'speaker': 'B', 'dia_id': 'D1:2', 'text': 'I like football'}],
            'session_2': [{'speaker': 'A', 'dia_id': 'D2:1', 'text': 'Jazz concert Friday'}]},
            'qa': [{'question': 'What music does A like?', 'answer': 'jazz',
                    'category': 1, 'evidence': ['D1:1; D2:1']},
                   {'question': 'Adversarial?', 'category': 5, 'evidence': []}]}]
        self.assertEqual(len(messages(fixture[0])), 3)
        self.assertEqual(evidence_ids(fixture[0]['qa'][0]), {'D1:1', 'D2:1'})
        results = run(fixture, k=3, retain=.67)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r['evidence_recall'] <= 1 for r in results))

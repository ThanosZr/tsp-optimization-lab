import itertools
import math
import tempfile
import unittest
from pathlib import Path

import bootstrap  # noqa: F401
from core.data import make_nodes
from core.models import tour_length
from core.session import Session


class WorkbenchTests(unittest.TestCase):
    def assert_tour(self, nodes, tour):
        self.assertCountEqual(tour.order, [node.idx for node in nodes])
        self.assertTrue(math.isfinite(tour.length))
        self.assertAlmostEqual(tour.length, tour_length(nodes, tour.order), places=9)

    def test_custom_tours(self):
        for mode in ('uniform', 'clustered', 'circle'):
            for seed in range(5):
                for n in (2, 3, 12, 30):
                    with self.subTest(mode=mode, seed=seed, n=n):
                        nodes = make_nodes(n, seed=seed, mode=mode)
                        algorithm = Session(nodes, seed=seed, data_mode=mode).algorithms['custom']
                        algorithm.run_all()
                        self.assertIsNotNone(algorithm.tour)
                        self.assert_tour(nodes, algorithm.tour)

    def test_insufficient_nodes_stop(self):
        for n in (0, 1):
            algorithm = Session(make_nodes(n)).algorithms["custom"]
            frames = algorithm.run_all()
            self.assertLess(len(frames), 10)
            self.assertFalse(algorithm.is_done())

    def test_som_and_exact(self):
        nodes = make_nodes(8, seed=7)
        session = Session(nodes)
        algorithm = session.algorithms['som']
        algorithm.run_all()
        for tour in algorithm.tours.values():
            self.assert_tour(nodes, tour)
        exact = algorithm.tours['exact']
        expected = min(tour_length(nodes, [0, *p]) for p in itertools.permutations(range(1, 8)))
        self.assertAlmostEqual(exact.length, expected, places=9)

    def test_session_round_trip(self):
        nodes = make_nodes(12, seed=7)
        session = Session(nodes)
        for algorithm in session.algorithms.values():
            algorithm.run_all()
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'session.json')
            session.save(path)
            loaded = Session.load(path)
        self.assertEqual(session.nodes, loaded.nodes)
        self.assertEqual(session.algorithms['custom'].tour.to_dict(), loaded.algorithms['custom'].tour.to_dict())
        self.assertEqual(session.algorithms['som'].tours['exact'].to_dict(), loaded.algorithms['som'].tours['exact'].to_dict())


if __name__ == '__main__':
    unittest.main()

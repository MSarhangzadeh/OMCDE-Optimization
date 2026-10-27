"""Run with: python -m unittest discover -s tests -v"""
import unittest
from unittest.mock import patch
import numpy as np
from omcde import _competitive_mutants, omcde

def sphere(x):
    return np.sum(x**2, axis=1)

class OMCDETests(unittest.TestCase):
    def test_budget_counts_opposition_and_last_partial_batch(self):
        count = [0]
        def objective(x):
            count[0] += len(x)
            return sphere(x)
        r = omcde(objective, dim=3, bounds=(-3, 3), max_evals=113,
                  population_size=14, seed=11)
        self.assertEqual(r.nfev, 113)
        self.assertEqual(count[0], 113)
        self.assertEqual(r.history['nfev'][0], 28)
        self.assertEqual(r.history['nfev'][-1], 113)

    def test_seed_reproducibility(self):
        a = omcde(sphere, dim=4, max_evals=650, seed=123)
        b = omcde(sphere, dim=4, max_evals=650, seed=123)
        self.assertTrue(np.array_equal(a.x, b.x))
        self.assertEqual(a.fun, b.fun)
        self.assertEqual(a.history['best_f'], b.history['best_f'])

    def test_non_multiple_of_three_population_sizes(self):
        for n in (4, 5, 7, 8, 10, 11):
            with self.subTest(n=n):
                r = omcde(sphere, dim=2, bounds=(-2, 2), max_evals=91,
                          population_size=n, seed=10)
                self.assertEqual(r.nfev, 91)
                self.assertEqual(len(r.final_population), 4)
                self.assertTrue(np.all(np.isfinite(r.final_population)))

    def test_best_is_monotone_and_within_bounds(self):
        r = omcde(sphere, dim=7, bounds=(-2, 3), max_evals=1000, seed=44)
        self.assertTrue(np.all(np.diff(r.history['best_f']) <= 1e-14))
        self.assertTrue(np.all(r.x >= -2))
        self.assertTrue(np.all(r.x <= 3))
        self.assertTrue(np.all(r.final_population >= -2))
        self.assertTrue(np.all(r.final_population <= 3))
        self.assertAlmostEqual(r.fun, float(sphere(r.x[None, :])[0]))
        self.assertIn('greedy', r.history['phase'])
        self.assertIn('competitive', r.history['phase'])

    def test_competitive_and_greedy_formulas(self):
        pop = np.array([[1., 1.], [2., 2.], [5., 5.]])
        fit = sphere(pop)
        factors = np.array([0.3, 0.4, 0.5])
        archive = np.empty((0, 2))
        gb = np.array([0., 0.])
        zero_difference = np.zeros((3, 2))
        with patch('omcde._random_differences', return_value=zero_difference):
            early = _competitive_mutants(
                np.random.default_rng(4), pop, fit, factors, archive, gb, False)
            late = _competitive_mutants(
                np.random.default_rng(4), pop, fit, factors, archive, gb, True)
        best, med, worst = pop
        np.testing.assert_allclose(early[0], best)
        np.testing.assert_allclose(early[1], med + .4 * (best - med))
        np.testing.assert_allclose(
            early[2], worst + .5 * (best - worst) + .5 * (med - worst))
        np.testing.assert_allclose(late[0], best + .3 * (gb - best))
        np.testing.assert_allclose(
            late[1], med + .4 * (gb - med) + .4 * (best - med))
        np.testing.assert_allclose(
            late[2], worst + .5 * (gb - worst)
            + .5 * (best - worst) + .5 * (med - worst))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '2 \\* initial'):
            omcde(sphere, dim=4, max_evals=10, seed=0)
        with self.assertRaisesRegex(ValueError, 'objective must map'):
            omcde(lambda x: 1.0, dim=2, max_evals=100, seed=0)
        with self.assertRaisesRegex(ValueError, 'lower bound'):
            omcde(sphere, dim=2, bounds=(5, 1), max_evals=100, seed=0)

    def test_sphere_optimization_progress(self):
        r = omcde(sphere, dim=5, bounds=(-5, 5), max_evals=3000, seed=42)
        self.assertLess(r.fun, 1e-5)

if __name__ == '__main__':
    unittest.main()

"""AI6 federated-learning attack engine tests — real code paths, offline."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_attack import (  # noqa: E402
    ByzantineDetector,
    FederatedClient,
    FederatedServer,
    FreeRider,
    GradientInverter,
    ModelUpdatePoisoner,
    run_experiment,
    test_client,
)


class TestFederatedPrimitives(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.input_size = 8
        self.num_classes = 3
        self.server = FederatedServer(self.input_size, self.num_classes, 4)
        self.clients = [FederatedClient(i, self.input_size, self.num_classes,
                                        data_size=40, seed=1 + i)
                        for i in range(4)]

    def test_client_local_train_returns_updates(self):
        uw, ub = self.clients[0].local_train(epochs=2, lr=0.01)
        self.assertEqual(len(uw), 2)
        self.assertEqual(uw[0].shape[1], 32)

    def test_aggregate_mean_shape(self):
        updates_w = []
        updates_b = []
        for c in self.clients:
            uw, ub = c.local_train(epochs=1, lr=0.01)
            updates_w.append(uw)
            updates_b.append(ub)
        agg_w, agg_b = self.server.aggregate_mean(updates_w, updates_b)
        self.assertEqual(len(agg_w), 2)
        self.assertEqual(agg_w[0].shape, self.clients[0].weights[0].shape)

    def test_model_update_poisoning_flips_sign(self):
        rng = np.random.RandomState(0)
        uw = [rng.randn(8, 32)]
        ub = [rng.randn(1, 3)]
        poisoner = ModelUpdatePoisoner(scale_factor=5.0)
        pw, pb = poisoner.sign_flip_poison(uw, ub)
        np.testing.assert_allclose(pw[0], -uw[0] * 5.0)

    def test_byzantine_detector_finds_outlier(self):
        # z-score outlier detection needs a realistic pool of clients (>=6);
        # with 1 malicious + 5 honest the scaled update is flagged.
        np.random.seed(3)
        n = 6
        server = FederatedServer(self.input_size, self.num_classes, n)
        clients = [FederatedClient(i, self.input_size, self.num_classes,
                                   data_size=40, seed=3 + i) for i in range(n)]
        updates_w = []
        updates_b = []
        for i, c in enumerate(clients):
            uw, ub = c.local_train(epochs=1, lr=0.01)
            if i == 0:
                uw = [w * 100.0 for w in uw]
                ub = [b * 100.0 for b in ub]
            updates_w.append(uw)
            updates_b.append(ub)
        detector = ByzantineDetector()
        outliers, norms = detector.detect_outliers(updates_w, updates_b,
                                                   threshold=2.0)
        self.assertIn(0, outliers)


class TestRunExperiment(unittest.TestCase):
    def test_returns_structured_results(self):
        r = run_experiment(input_size=6, num_classes=3, num_clients=4,
                           num_rounds=2, seed=1)
        self.assertIn("baseline", r)
        self.assertIn("poisoning", r)
        self.assertIn("gradient_inversion", r)
        self.assertIn("free_rider", r)
        self.assertIn("defenses", r)
        self.assertIn("krum_accuracy", r["defenses"])
        self.assertIn("trimmed_mean_accuracy", r["defenses"])

    def test_deterministic_with_seed(self):
        a = run_experiment(input_size=6, num_classes=3, num_clients=4,
                           num_rounds=2, seed=7)
        b = run_experiment(input_size=6, num_classes=3, num_clients=4,
                           num_rounds=2, seed=7)
        self.assertEqual(a["baseline"]["accuracy"],
                         b["baseline"]["accuracy"])


class TestCLI(unittest.TestCase):
    def test_cli_writes_json_report_and_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.json")
            from fl_attack import main
            code = main(["--clients", "4", "--rounds", "2", "--features", "6",
                         "--seed", "1", "--output", out, "--quiet"])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["setup"]["num_clients"], 4)


if __name__ == "__main__":
    unittest.main()
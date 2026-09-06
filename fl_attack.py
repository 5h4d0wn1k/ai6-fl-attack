#!/usr/bin/env python3
"""
AI6 — Federated Learning Attack
Demonstrates model update poisoning, gradient inversion, free-rider attack,
and Byzantine tolerance bypass using only numpy.
"""


import numpy as np


class FederatedClient:
    """Simulated federated learning client with local data and model."""

    def __init__(self, client_id, input_size, num_classes, data_size=100, seed=42):
        self.client_id = client_id
        self.input_size = input_size
        self.num_classes = num_classes
        rng = np.random.RandomState(seed + client_id)

        self.X = rng.randn(data_size, input_size)
        self.y = rng.randint(0, num_classes, data_size)

        self.weights = []
        self.biases = []
        layer_sizes = [input_size, 32, num_classes]
        for i in range(len(layer_sizes) - 1):
            w = rng.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        z_values = []
        current = X
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            z_values.append(z)
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = self.softmax(z)
            activations.append(current)
        return activations, z_values

    def compute_loss(self, X, y):
        activations, _ = self.forward(X)
        probs = activations[-1]
        m = X.shape[0]
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(m), y] = 1
        eps = 1e-8
        loss = -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))
        return loss

    def local_train(self, epochs=5, lr=0.01):
        old_w = [w.copy() for w in self.weights]
        old_b = [b.copy() for b in self.biases]

        for _ in range(epochs):
            activations, z_values = self.forward(self.X)
            m = self.X.shape[0]
            one_hot = np.zeros_like(activations[-1])
            one_hot[np.arange(m), self.y] = 1
            delta = activations[-1] - one_hot

            grads_w = [None] * len(self.weights)
            grads_b = [None] * len(self.biases)
            grads_w[-1] = activations[-2].T @ delta / m
            grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

            for i in range(len(self.weights) - 2, -1, -1):
                delta = (delta @ self.weights[i + 1].T) * (z_values[i] > 0).astype(float)
                grads_w[i] = activations[i].T @ delta / m
                grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m

            for i in range(len(self.weights)):
                self.weights[i] -= lr * grads_w[i]
                self.biases[i] -= lr * grads_b[i]

        update_w = [old_w[i] - self.weights[i] for i in range(len(self.weights))]
        update_b = [old_b[i] - self.biases[i] for i in range(len(self.biases))]
        return update_w, update_b

    def apply_update(self, update_w, update_b):
        for i in range(len(self.weights)):
            self.weights[i] += update_w[i]
            self.biases[i] += update_b[i]

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)


class FederatedServer:
    """Federated learning server with aggregation strategies."""

    def __init__(self, input_size, num_classes, num_clients):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_clients = num_clients
        layer_sizes = [input_size, 32, num_classes]
        rng = np.random.RandomState(0)
        self.global_weights = []
        self.global_biases = []
        for i in range(len(layer_sizes) - 1):
            w = rng.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, layer_sizes[i + 1]))
            self.global_weights.append(w)
            self.global_biases.append(b)
        self.history = []

    def aggregate_mean(self, updates_w, updates_b, weights=None):
        n = len(updates_w)
        if weights is None:
            weights = np.ones(n) / n
        agg_w = [np.zeros_like(updates_w[0][i]) for i in range(len(updates_w[0]))]
        agg_b = [np.zeros_like(updates_b[0][i]) for i in range(len(updates_b[0]))]
        for k in range(n):
            for i in range(len(agg_w)):
                agg_w[i] += weights[k] * updates_w[k][i]
                agg_b[i] += weights[k] * updates_b[k][i]
        return agg_w, agg_b

    def aggregate_krum(self, updates_w, updates_b, Byzantine_count=1):
        n = len(updates_w)
        flat_updates = []
        for k in range(n):
            flat = np.concatenate([w.flatten() for w in updates_w[k]] +
                                  [b.flatten() for b in updates_b[k]])
            flat_updates.append(flat)
        flat_updates = np.array(flat_updates)

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(flat_updates[i] - flat_updates[j])
                distances[i, j] = d
                distances[j, i] = d

        f = Byzantine_count
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:n - f])
            scores.append(score)
        scores = np.array(scores)

        best_idx = np.argmin(scores)
        return updates_w[best_idx], updates_b[best_idx]

    def aggregate_trimmed_mean(self, updates_w, updates_b, trim_ratio=0.1):
        n = len(updates_w)
        n_layers_w = len(updates_w[0])
        n_layers_b = len(updates_b[0])

        agg_w = []
        agg_b = []

        for layer_idx in range(n_layers_w):
            stacked = np.array([updates_w[k][layer_idx] for k in range(n)])
            trim = max(1, int(n * trim_ratio))
            sorted_vals = np.sort(stacked, axis=0)
            trimmed = sorted_vals[trim:-trim] if trim < n - trim else sorted_vals
            agg_w.append(np.mean(trimmed, axis=0))

        for layer_idx in range(n_layers_b):
            stacked = np.array([updates_b[k][layer_idx] for k in range(n)])
            trim = max(1, int(n * trim_ratio))
            sorted_vals = np.sort(stacked, axis=0)
            trimmed = sorted_vals[trim:-trim] if trim < n - trim else sorted_vals
            agg_b.append(np.mean(trimmed, axis=0))

        return agg_w, agg_b

    def set_global_model(self, weights, biases):
        self.global_weights = [w.copy() for w in weights]
        self.global_biases = [b.copy() for b in biases]

    def broadcast(self):
        return [w.copy() for w in self.global_weights], [b.copy() for b in self.global_biases]


class ModelUpdatePoisoner:
    """Poisons model updates from compromised clients."""

    def __init__(self, scale_factor=10.0, flip_sign=True):
        self.scale_factor = scale_factor
        self.flip_sign = flip_sign

    def scale_poison(self, update_w, update_b):
        poisoned_w = [w * self.scale_factor for w in update_w]
        poisoned_b = [b * self.scale_factor for b in update_b]
        return poisoned_w, poisoned_b

    def sign_flip_poison(self, update_w, update_b):
        poisoned_w = [-w * self.scale_factor for w in update_w]
        poisoned_b = [-b * self.scale_factor for b in update_b]
        return poisoned_w, poisoned_b

    def gaussian_noise_poison(self, update_w, update_b, noise_scale=5.0, seed=42):
        rng = np.random.RandomState(seed)
        poisoned_w = [w + rng.randn(*w.shape) * noise_scale for w in update_w]
        poisoned_b = [b + rng.randn(*b.shape) * noise_scale for b in update_b]
        return poisoned_w, poisoned_b

    def label_flipping_poison(self, update_w, update_b, num_classes):
        poisoned_w = [-w for w in update_w]
        poisoned_b = [-b for b in update_b]
        for i in range(len(poisoned_b)):
            shifted = np.roll(poisoned_b[i], 1, axis=1)
            poisoned_b[i] = shifted
        return poisoned_w, poisoned_b


class GradientInverter:
    """Inverts model updates to recover training data."""

    def __init__(self, input_size, num_classes, max_iters=500, lr=0.01):
        self.input_size = input_size
        self.num_classes = num_classes
        self.max_iters = max_iters
        self.lr = lr

    def invert(self, update_w, update_b, target_label=None):
        X_recovered = np.random.randn(1, self.input_size) * 0.01
        X_recovered = X_recovered.astype(np.float64)

        if target_label is None:
            target_label = 0

        for iteration in range(self.max_iters):

            current = X_recovered
            activations = [current]
            z_values = []

            wfake = [u.copy() for u in update_w]
            bfake = [u.copy() for u in update_b]

            for i in range(len(wfake)):
                z = current @ wfake[i] + bfake[i]
                z_values.append(z)
                if i < len(wfake) - 1:
                    current = np.maximum(0, z)
                else:
                    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                    current = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                activations.append(current)

            probs = activations[-1]
            target_onehot = np.zeros((1, self.num_classes))
            target_onehot[0, target_label] = 1.0

            loss_grad = probs - target_onehot

            delta = loss_grad @ wfake[-1].T
            if len(z_values) > 1:
                delta = delta * (z_values[-2] > 0).astype(float)
            X_grad = delta @ wfake[0].T

            X_recovered -= self.lr * X_grad

            norm = np.linalg.norm(X_recovered)
            if norm > 10:
                X_recovered = X_recovered / norm * 10

        return X_recovered

    def batch_invert(self, update_w, update_b, batch_size=5):
        recovered = []
        for label in range(min(batch_size, self.num_classes)):
            X = self.invert(update_w, update_b, target_label=label)
            recovered.append(X)
        return np.vstack(recovered)


class FreeRider:
    """Free-rider attack: contributes nothing but benefits from aggregation."""

    def __init__(self, input_size, num_classes, num_layers=2):
        self.input_size = input_size
        self.num_classes = num_classes
        layer_sizes = [input_size, 32, num_classes]
        self.zeros_w = [np.zeros((layer_sizes[i], layer_sizes[i + 1]))
                        for i in range(len(layer_sizes) - 1)]
        self.zeros_b = [np.zeros((1, layer_sizes[i + 1]))
                        for i in range(len(layer_sizes) - 1)]

    def generate_fake_update(self, magnitude=0.001, seed=42):
        rng = np.random.RandomState(seed)
        fake_w = [w + rng.randn(*w.shape) * magnitude for w in self.zeros_w]
        fake_b = [b + rng.randn(*b.shape) * magnitude for b in self.zeros_b]
        return fake_w, fake_b

    def mimicking_update(self, other_updates_w, other_updates_b, noise_scale=0.01):
        if len(other_updates_w) == 0:
            return self.zeros_w, self.zeros_b
        rng = np.random.RandomState(42)
        mimic_w = []
        mimic_b = []
        for i in range(len(self.zeros_w)):
            avg_w = np.mean([u[i] for u in other_updates_w], axis=0)
            mimic_w.append(avg_w + rng.randn(*avg_w.shape) * noise_scale)
        for i in range(len(self.zeros_b)):
            avg_b = np.mean([u[i] for u in other_updates_b], axis=0)
            mimic_b.append(avg_b + rng.randn(*avg_b.shape) * noise_scale)
        return mimic_w, mimic_b


class ByzantineDetector:
    """Detects Byzantine (malicious) clients."""

    def __init__(self):
        self.scores_history = []

    def compute_update_norms(self, updates_w, updates_b):
        norms = []
        for k in range(len(updates_w)):
            flat = np.concatenate([w.flatten() for w in updates_w[k]] +
                                  [b.flatten() for b in updates_b[k]])
            norms.append(np.linalg.norm(flat))
        return np.array(norms)

    def detect_outliers(self, updates_w, updates_b, threshold=2.0):
        norms = self.compute_update_norms(updates_w, updates_b)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        if std_norm == 0:
            return list(range(len(norms))), norms

        z_scores = (norms - mean_norm) / std_norm
        outlier_indices = np.where(np.abs(z_scores) > threshold)[0].tolist()
        self.scores_history.append(z_scores)
        return outlier_indices, norms

    def filter_byzantine(self, updates_w, updates_b, clients, threshold=2.0):
        outlier_ids, norms = self.detect_outliers(updates_w, updates_b, threshold)
        filtered_w = []
        filtered_b = []
        filtered_clients = []
        for k in range(len(updates_w)):
            if k not in outlier_ids:
                filtered_w.append(updates_w[k])
                filtered_b.append(updates_b[k])
                filtered_clients.append(clients[k])
        return filtered_w, filtered_b, filtered_clients, outlier_ids


def accuracy(model_func, X, y):
    preds = model_func(X)
    return np.mean(preds == y)


def test_client(client, X_test, y_test):
    return accuracy(client.predict, X_test, y_test)


def run_experiment(input_size: int = 10, num_classes: int = 3,
                   num_clients: int = 6, num_rounds: int = 5,
                   seed: int = 42) -> dict:
    """Run the full federated learning attack experiment, returning results."""
    np.random.seed(seed)
    clients = [FederatedClient(i, input_size, num_classes, data_size=150,
                               seed=seed + i)
               for i in range(num_clients)]
    server = FederatedServer(input_size, num_classes, num_clients)

    X_test = np.random.randn(200, input_size)
    y_test = np.random.RandomState(99).randint(0, num_classes, 200)

    # [1] Baseline
    for _ in range(num_rounds):
        global_w, global_b = server.broadcast()
        updates_w, updates_b = [], []
        for client in clients:
            client.weights = [w.copy() for w in global_w]
            client.biases = [b.copy() for b in global_b]
            uw, ub = client.local_train(epochs=5, lr=0.01)
            updates_w.append(uw)
            updates_b.append(ub)
        agg_w, agg_b = server.aggregate_mean(updates_w, updates_b)
        server.set_global_model(agg_w, agg_b)
        for client in clients:
            client.weights = [w.copy() for w in server.global_weights]
            client.biases = [b.copy() for b in server.global_biases]

    acc_baseline = float(test_client(clients[0], X_test, y_test))

    # [2] Model update poisoning (sign flip)
    server2 = FederatedServer(input_size, num_classes, num_clients)
    clients2 = [FederatedClient(i, input_size, num_classes, data_size=150,
                                seed=seed + i)
                for i in range(num_clients)]
    poisoner = ModelUpdatePoisoner(scale_factor=5.0)

    for _ in range(num_rounds):
        global_w, global_b = server2.broadcast()
        updates_w, updates_b = [], []
        for i, client in enumerate(clients2):
            client.weights = [w.copy() for w in global_w]
            client.biases = [b.copy() for b in global_b]
            uw, ub = client.local_train(epochs=5, lr=0.01)
            if i == 0:
                uw, ub = poisoner.sign_flip_poison(uw, ub)
            updates_w.append(uw)
            updates_b.append(ub)
        agg_w, agg_b = server2.aggregate_mean(updates_w, updates_b)
        server2.set_global_model(agg_w, agg_b)
        for client in clients2:
            client.weights = [w.copy() for w in server2.global_weights]
            client.biases = [b.copy() for b in server2.global_biases]

    acc_poisoned = float(test_client(clients2[0], X_test, y_test))

    # [3] Gradient inversion
    gradient_inv = GradientInverter(input_size, num_classes, max_iters=300, lr=0.05)
    target_client = clients[0]
    backup_w = [w.copy() for w in target_client.weights]
    backup_b = [b.copy() for b in target_client.biases]
    global_w_ref = [w.copy() for w in server.global_weights]
    global_b_ref = [b.copy() for b in server.global_biases]
    target_client.weights = [w.copy() for w in global_w_ref]
    target_client.biases = [b.copy() for b in global_b_ref]
    real_update_w, real_update_b = target_client.local_train(epochs=3, lr=0.01)
    target_client.weights = backup_w
    target_client.biases = backup_b

    X_inv = gradient_inv.invert(real_update_w, real_update_b, target_label=0)
    X_batch = gradient_inv.batch_invert(real_update_w, real_update_b, batch_size=3)

    # [4] Free-rider attack
    server3 = FederatedServer(input_size, num_classes, num_clients)
    clients3 = [FederatedClient(i, input_size, num_classes, data_size=150,
                                seed=seed + i)
                for i in range(num_clients)]
    free_rider = FreeRider(input_size, num_classes)

    for _ in range(num_rounds):
        global_w, global_b = server3.broadcast()
        updates_w, updates_b = [], []
        for i, client in enumerate(clients3):
            client.weights = [w.copy() for w in global_w]
            client.biases = [b.copy() for b in global_b]
            if i < num_clients - 1:
                uw, ub = client.local_train(epochs=5, lr=0.01)
            else:
                uw, ub = free_rider.generate_fake_update(magnitude=0.0001)
            updates_w.append(uw)
            updates_b.append(ub)
        agg_w, agg_b = server3.aggregate_mean(updates_w, updates_b)
        server3.set_global_model(agg_w, agg_b)
        for client in clients3:
            client.weights = [w.copy() for w in server3.global_weights]
            client.biases = [b.copy() for b in server3.global_biases]

    honest_client_acc = float(test_client(clients3[0], X_test, y_test))
    fr_client = clients3[-1]
    fr_client.weights = [w.copy() for w in server3.global_weights]
    fr_client.biases = [b.copy() for b in server3.global_biases]
    acc_freerider_benefit = float(test_client(fr_client, X_test, y_test))

    # [5] Byzantine tolerance bypass (Krum)
    server4 = FederatedServer(input_size, num_classes, num_clients)
    clients4 = [FederatedClient(i, input_size, num_classes, data_size=150,
                                seed=seed + i)
                for i in range(num_clients)]
    poisoner2 = ModelUpdatePoisoner(scale_factor=10.0)
    detector = ByzantineDetector()

    for _ in range(num_rounds):
        global_w, global_b = server4.broadcast()
        updates_w, updates_b = [], []
        for i, client in enumerate(clients4):
            client.weights = [w.copy() for w in global_w]
            client.biases = [b.copy() for b in global_b]
            uw, ub = client.local_train(epochs=5, lr=0.01)
            if i == 0:
                uw, ub = poisoner2.scale_poison(uw, ub)
            updates_w.append(uw)
            updates_b.append(ub)

        outlier_ids, _ = detector.detect_outliers(updates_w, updates_b,
                                                  threshold=2.0)
        if len(outlier_ids) > 0 and len(outlier_ids) < len(updates_w):
            clean_w = [updates_w[i] for i in range(len(updates_w))
                       if i not in outlier_ids]
            clean_b = [updates_b[i] for i in range(len(updates_b))
                       if i not in outlier_ids]
        else:
            clean_w = updates_w
            clean_b = updates_b

        agg_w, agg_b = server4.aggregate_krum(clean_w, clean_b, Byzantine_count=1)
        server4.set_global_model(agg_w, agg_b)
        for client in clients4:
            client.weights = [w.copy() for w in server4.global_weights]
            client.biases = [b.copy() for b in server4.global_biases]

    acc_krum = float(test_client(clients4[0], X_test, y_test))

    # [6] Trimmed mean defense
    server5 = FederatedServer(input_size, num_classes, num_clients)
    clients5 = [FederatedClient(i, input_size, num_classes, data_size=150,
                                seed=seed + i)
                for i in range(num_clients)]

    for _ in range(num_rounds):
        global_w, global_b = server5.broadcast()
        updates_w, updates_b = [], []
        for i, client in enumerate(clients5):
            client.weights = [w.copy() for w in global_w]
            client.biases = [b.copy() for b in global_b]
            uw, ub = client.local_train(epochs=5, lr=0.01)
            if i == 0:
                uw, ub = poisoner2.sign_flip_poison(uw, ub)
            updates_w.append(uw)
            updates_b.append(ub)
        agg_w, agg_b = server5.aggregate_trimmed_mean(updates_w, updates_b,
                                                      trim_ratio=0.2)
        server5.set_global_model(agg_w, agg_b)
        for client in clients5:
            client.weights = [w.copy() for w in server5.global_weights]
            client.biases = [b.copy() for b in server5.global_biases]

    acc_trimmed = float(test_client(clients5[0], X_test, y_test))

    return {
        "setup": {
            "input_size": input_size,
            "num_classes": num_classes,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "seed": seed,
        },
        "baseline": {"accuracy": acc_baseline},
        "poisoning": {
            "accuracy_under_attack": acc_poisoned,
            "accuracy_drop": acc_baseline - acc_poisoned,
        },
        "gradient_inversion": {
            "inverted_sample_shape": list(X_inv.shape),
            "inverted_norm": float(np.linalg.norm(X_inv)),
            "batch_inverted_shape": list(X_batch.shape),
        },
        "free_rider": {
            "honest_client_accuracy": honest_client_acc,
            "free_rider_accuracy": acc_freerider_benefit,
            "free_rider_benefit": bool(
                acc_freerider_benefit >= honest_client_acc * 0.9),
        },
        "defenses": {
            "krum_accuracy": acc_krum,
            "trimmed_mean_accuracy": acc_trimmed,
            "krum_improvement_over_poisoned": acc_krum - acc_poisoned,
            "trimmed_improvement_over_poisoned": acc_trimmed - acc_poisoned,
        },
        "summary": {
            "baseline": acc_baseline,
            "under_poisoning": acc_poisoned,
            "krum_defense": acc_krum,
            "trimmed_defense": acc_trimmed,
            "free_rider_benefit": acc_freerider_benefit,
        },
    }


def format_report(results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("AI6 — Federated Learning Attack Demonstration")
    lines.append("=" * 60)

    lines.append("\n[1] Baseline federated learning (no attacks)...")
    lines.append(f"    Baseline accuracy: {results['baseline']['accuracy']:.4f}")

    lines.append("\n[2] Model update poisoning (sign flip attack)...")
    p = results["poisoning"]
    lines.append(f"    Accuracy under poisoning: {p['accuracy_under_attack']:.4f}")
    lines.append(f"    Accuracy drop: {p['accuracy_drop']:+.4f}")

    lines.append("\n[3] Gradient inversion attack...")
    gi = results["gradient_inversion"]
    lines.append(f"    Inverted sample shape:  {gi['inverted_sample_shape']}")
    lines.append(f"    Recovered gradient norm: {gi['inverted_norm']:.4f}")
    lines.append(f"    Batch inversion shape:  {gi['batch_inverted_shape']}")

    lines.append("\n[4] Free-rider attack...")
    fr = results["free_rider"]
    lines.append(f"    Honest client accuracy: {fr['honest_client_accuracy']:.4f}")
    lines.append(f"    Free-rider accuracy:    {fr['free_rider_accuracy']:.4f}")
    lines.append(f"    Free-rider got benefit: {fr['free_rider_benefit']}")

    lines.append("\n[5] Byzantine tolerance bypass (Krum aggregation)...")
    d = results["defenses"]
    lines.append(f"    Krum accuracy (with detection): {d['krum_accuracy']:.4f}")
    lines.append(f"    Improvement: {d['krum_improvement_over_poisoned']:+.4f}")

    lines.append("\n[6] Trimmed mean aggregation defense...")
    lines.append(f"    Trimmed mean accuracy: {d['trimmed_mean_accuracy']:.4f}")

    lines.append("\n" + "=" * 60)
    lines.append("Results Summary:")
    for k, v in results["summary"].items():
        lines.append(f"  {k.replace('_', ' ').title():<22}{v:.4f}")
    lines.append("=" * 60)
    lines.append("Demonstration complete.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        prog="ai6-fl-attack",
        description="Federated learning attack research: update poisoning, "
                    "gradient inversion, free-rider, Byzantine bypass. "
                    "Offline, self-contained.")
    parser.add_argument("--clients", type=int, default=6,
                        help="number of simulated clients")
    parser.add_argument("--rounds", type=int, default=5,
                        help="federated rounds per scenario")
    parser.add_argument("--features", type=int, default=10,
                        help="input feature count")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", metavar="FILE",
                        help="write JSON report to FILE (e.g. reports/ai6-report.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress human-readable output")
    args = parser.parse_args(argv)

    results = run_experiment(input_size=args.features,
                             num_clients=args.clients,
                             num_rounds=args.rounds, seed=args.seed)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    if not args.quiet:
        print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

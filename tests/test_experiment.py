"""Run with: python -m unittest discover -s tests -v"""

from contextlib import redirect_stdout
import copy
import io
import os
from pathlib import Path
import random
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

import experiment


class ExperimentTests(unittest.TestCase):
    def test_config_and_paths_are_independent_of_working_directory(self):
        previous = Path.cwd()
        expected = experiment.load_config()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                config = experiment.load_config()
                self.assertEqual(config, expected)
                self.assertIsInstance(config["contrastive"]["learning_rate"], float)
                config["paths"]["iu_xray_dir"] = "data/iu_xray"
                self.assertEqual(
                    experiment.project_path(config, "iu_xray_dir"),
                    experiment.PROJECT_ROOT / "data" / "iu_xray",
                )
            finally:
                os.chdir(previous)

    def test_invalid_seeds_fail_when_loading(self):
        config = experiment.load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            for seed in [True, -1, 2**32, "42"]:
                with self.subTest(seed=seed):
                    config["seed"] = seed
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "seed"):
                        experiment.load_config(path)

    def test_snapshot_can_be_loaded_again(self):
        config = experiment.load_config()
        config["seed"] = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "snapshot.yaml"
            experiment.save_config(config, path)
            self.assertEqual(experiment.load_config(path), config)

    def test_python_numpy_and_torch_repeat_after_reseeding(self):
        def draw(seed):
            experiment.seed_everything(seed)
            return random.random(), np.random.rand(5), torch.rand(5)

        first = draw(42)
        second = draw(42)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertFalse(torch.equal(first[2], draw(7)[2]))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_loader_shuffle_is_independent_of_global_rng_and_other_loaders(self):
        config = experiment.load_config()
        config["seed"] = 42
        dataset = TensorDataset(torch.arange(32))

        def order():
            loader = DataLoader(
                dataset, batch_size=8, shuffle=True,
                **experiment.dataloader_options(config),
            )
            return torch.cat([batch[0] for batch in loader])

        expected = order()
        torch.rand(100)
        order()  # Iterating another loader must not change a new loader's order.
        self.assertTrue(torch.equal(expected, order()))
        config["seed"] = 7
        self.assertFalse(torch.equal(expected, order()))

    def test_windows_workers_fail_with_an_actionable_message(self):
        config = experiment.load_config()
        config["runtime"]["num_workers"] = 2
        with patch.object(experiment.os, "name", "nt"):
            with self.assertRaisesRegex(ValueError, "__main__"):
                experiment.dataloader_options(config)

    def test_basic_training_uses_configuration_and_is_repeatable(self):
        config = copy.deepcopy(experiment.load_config())
        config["runtime"]["device"] = "cpu"
        config["basics"].update({
            "samples": 160, "input_dim": 10, "hidden_dim": 8,
            "batch_size": 16, "epochs": 1, "learning_rate": 0.0123,
        })

        def run():
            with patch.object(experiment, "load_config", return_value=config):
                with redirect_stdout(io.StringIO()):
                    return runpy.run_path(
                        str(experiment.PROJECT_ROOT / "01_pytorch_basics.py")
                    )

        first = run()
        second = run()
        self.assertEqual(first["X"].shape, (160, 10))
        self.assertEqual(first["model"][0].out_features, 8)
        self.assertEqual(first["train_loader"].batch_size, 16)
        self.assertEqual(first["optimizer"].param_groups[0]["lr"], 0.0123)
        self.assertEqual(first["epoch"], 0)
        for name, weight in first["model"].state_dict().items():
            self.assertTrue(torch.equal(weight, second["model"].state_dict()[name]))


if __name__ == "__main__":
    unittest.main()

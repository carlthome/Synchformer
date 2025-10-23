import torch

from dataset.transforms import make_class_grid


def test_make_class_grid():
    grid = make_class_grid(-1.0, 1.0, 5)
    expected_grid = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert torch.allclose(grid, expected_grid)

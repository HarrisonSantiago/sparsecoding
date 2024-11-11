import torch
import unittest

from transforms import whiten

class TestWhitener(unittest.TestCase):
    def test_pca(self):
        N = 100
        D = 10

        torch.manual_seed(42)

        data = torch.rand((N, D), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'pca')

        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(D, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
            torch.cov(whitened_data.T),
            torch.eye(D, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        data = torch.Tensor([[1, 0], [2, 0]])
        whitened_data = whiten(data, algorithm = 'pca')

        assert not torch.any(torch.isnan(
            whitened_data,
        )), "If an eigenvalue is 0, should not get NaNs."


    def test_zca(self):
        N = 100
        D = 10

        torch.manual_seed(42)

        data = torch.rand((N, D), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'zca')

        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(D, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
            torch.cov(whitened_data.T),
            torch.eye(D, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        data = torch.Tensor([[1, 0], [2, 0]])
        whitened_data = whiten(data, algorithm = 'zca')

        assert not torch.any(torch.isnan(
            whitened_data,
        )), "If an eigenvalue is 0, should not get NaNs."

    def test_cholesky(self):
        N = 100
        D = 10

        torch.manual_seed(42)

        data = torch.rand((N, D), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'cholesky')

        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(D, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
            torch.cov(whitened_data.T),
            torch.eye(D, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        data = torch.Tensor([[1, 0], [2, 0]])
        whitened_data = whiten(data, algorithm = 'cholesky')

        assert not torch.any(torch.isnan(
            whitened_data,
        )), "If an eigenvalue is 0, should not get NaNs."


if __name__ == "__main__":
    unittest.main()

import torch
import unittest

from sparsecoding.transforms import whiten

class TestWhitener(unittest.TestCase):
    def test_pca(self):
        N = 400
        C = 1
        H = 16
        W = 16

        torch.manual_seed(42)

        data = torch.rand((N, C, H, W), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'pca')

        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(1, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
            torch.cov(whitened_data.reshape(N, -1).T),
            torch.eye(C * H * W, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        data = torch.Tensor([[1, 0], [2, 0]])
        whitened_data = whiten(data, algorithm = 'pca')

        assert not torch.any(torch.isnan(
            whitened_data,
        )), "If an eigenvalue is 0, should not get NaNs."


    def test_zca(self):
        N = 400
        C = 1
        H = 16
        W = 16

        torch.manual_seed(42)

        data = torch.rand((N, C, H, W), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'zca', epsilon = 1e-2)

        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(1, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
            torch.cov(whitened_data.reshape(N, -1).T),
            torch.eye(C * H * W, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        data = torch.Tensor([[1, 0], [2, 0]])
        whitened_data = whiten(data, algorithm = 'zca')

        assert not torch.any(torch.isnan(
            whitened_data,
        )), "If an eigenvalue is 0, should not get NaNs."

    def test_cholesky(self):
        N = 400
        C = 1
        H = 16
        W = 16

        torch.manual_seed(42)

        data = torch.rand((N, C, H, W), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'cholesky')
        print(torch.cov(whitened_data.reshape(N, -1)))
        
        assert torch.allclose(
            torch.mean(whitened_data),
            torch.zeros(1, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
            torch.cov(whitened_data.reshape(N, -1).T),
            torch.eye(C * H * W, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        data = torch.Tensor([[1, 0], [2, 0]])
        whitened_data = whiten(data, algorithm = 'cholesky')

        assert not torch.any(torch.isnan(
            whitened_data,
        )), "If an eigenvalue is 0, should not get NaNs."


if __name__ == "__main__":
    unittest.main()

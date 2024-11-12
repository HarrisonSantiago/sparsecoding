import torch
import unittest

from sparsecoding.transforms import whiten

class TestWhitener(unittest.TestCase):
    def test_pca(self):
        N = 6000
        C = 1
        H = 28
        W = 28

        torch.manual_seed(42)

        data = torch.randn((N, C, H, W), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'pca')

        y_cov = torch.cov(whitened_data.reshape(N, -1).T)
        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(1, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
             y_cov,
            torch.eye(C * H * W, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

       

    def test_zca(self):
        N = 6000
        C = 1
        H = 28
        W = 28

        torch.manual_seed(42)

        data = torch.randn((N, C, H, W), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'zca')

        y_cov = torch.cov(whitened_data.reshape(N, -1).T)
        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(1, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
             y_cov,
            torch.eye(C * H * W, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        
    def test_cholesky(self):
        N = 6000
        C = 1
        H = 28
        W = 28

        torch.manual_seed(42)

        data = torch.randn((N, C, H, W), dtype=torch.float32)
        whitened_data = whiten(data, algorithm = 'cholesky')
        
        y_cov = torch.cov(whitened_data.reshape(N, -1).T)
        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(1, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
             y_cov,
            torch.eye(C * H * W, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        

if __name__ == "__main__":
    unittest.main()

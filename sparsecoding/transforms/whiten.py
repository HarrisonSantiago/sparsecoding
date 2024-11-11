import torch
import torch.fft as fft
import numpy as np
from typing import Dict, Optional
from functools import lru_cache


def create_frequency_filter(image_size: int, f0_factor: float = 0.4) -> torch.Tensor:
    """
    Create a frequency domain filter for image whitening.
    
    Args:
        image_size: Size of the square image
        f0_factor: Factor for determining the cutoff frequency (default 0.4)
        
    Returns:
        torch.Tensor: Frequency domain filter
    """
    fx = torch.linspace(-image_size/2, image_size/2-1, image_size)
    fy = torch.linspace(-image_size/2, image_size/2-1, image_size)
    fx, fy = torch.meshgrid(fx, fy, indexing='xy')
    
    rho = torch.sqrt(fx**2 + fy**2)
    f_0 = f0_factor * image_size
    filt = rho * torch.exp(-(rho/f_0)**4)
    
    return fft.fftshift(filt)

@lru_cache(maxsize=32)
def get_cached_filter(image_size: int, f0_factor: float = 0.4) -> torch.Tensor:
    """
    Get a cached frequency filter for the given image size.
    
    Args:
        image_size: Size of the square image
        f0_factor: Factor for determining the cutoff frequency
        
    Returns:
        torch.Tensor: Cached frequency domain filter
    """
    return create_frequency_filter(image_size, f0_factor)

def normalize_variance(tensor: torch.Tensor, target_variance: float = 0.1) -> torch.Tensor:
    """
    Normalize the variance of a tensor to a target value.
    
    Args:
        tensor: Input tensor
        target_variance: Desired variance after normalization
        
    Returns:
        torch.Tensor: Normalized tensor
    """
    if torch.var(tensor) < 1e-8:
        return tensor
        
    centered = tensor - tensor.mean()
    current_variance = torch.var(centered)
    
    if current_variance > 0:
        scale_factor = torch.sqrt(torch.tensor(target_variance) / current_variance)
        return centered * scale_factor
    return centered

def whiten_channel(
    channel: torch.Tensor,
    filt: torch.Tensor,
    target_variance: float = 0.1
) -> torch.Tensor:
    """
    Apply frequency domain whitening to a single channel.
    
    Args:
        channel: Single channel image tensor
        filt: Frequency domain filter
        target_variance: Target variance for normalization
        
    Returns:
        torch.Tensor: Whitened channel
    """

    if torch.var(channel) < 1e-8:
        return channel
    
    # Convert to frequency domain and apply filter
    If = fft.fft2(channel)
    If_whitened = If * filt.to(channel.device)
    
    # Convert back to spatial domain and normalize
    whitened = torch.real(fft.ifft2(If_whitened))

    # Normalize variance
    whitened = whitened - whitened.mean()
    variance = torch.var(whitened)
    if variance > 0:  
        scale_factor = torch.sqrt(torch.tensor(target_variance) / variance)
        whitened = whitened * scale_factor

    return whitened

def frequency_whitening(
    images: torch.Tensor,
    target_variance: float = 0.1,
    f0_factor: float = 0.4
) -> torch.Tensor:
    """
    Apply frequency domain decorrelation to batched images.
    Method used in original sparsenet in Olshausen and Field in Nature
    and http://www.rctn.org/bruno/sparsenet/
    
    Args:
        images: Input images of shape (N, C, H, W)
        target_variance: Target variance for normalization
        f0_factor: Factor for determining filter cutoff frequency
        
    Returns:
        torch.Tensor: Whitened images
    """
    N, C, H, W = images.shape
    if H != W:
        raise ValueError("Images must be square")
    
    # Get cached filter
    filt = get_cached_filter(H, f0_factor)
    
    # Process each image in the batch
    whitened_batch = []
    for img in images:
        whitened_channels = [
            whiten_channel(img[c], filt, target_variance)
            for c in range(C)
        ]
        whitened_batch.append(torch.stack(whitened_channels))
    
    return torch.stack(whitened_batch)

def compute_whitening_stats(data: torch.Tensor, algorithm = 'zca', n_components = None, **kwargs) -> Dict:

    """
    Given a tensor of data, compute statistics for whitening transform.
  
    Args:
        data: Input data where samples are along first axis
        mode: One of ['zca', 'pca']
        n_components: Number of principal components to keep. If None, keep all components.
                        If int, keep that many components. If float between 0 and 1,
                        keep components that explain that fraction of variance.
        
    Returns:
        Dictionary containing PCA statistics and explained variance ratio
    """

    if algorithm not in ['zca', 'pca', 'cholesky']:
        raise ValueError(f"algorithm must be one of ['zca', 'pca'], got {algorithm}")

    data = data.reshape((-1, np.prod(data.size()[1:])))


    # Step 1: Compute mean
    mean = torch.mean(data, dim=0)
    Sigma = torch.cov(data.T)

    # Step 2: Compute eigenvalues/eigenvectors
    # We do this via SVD as it is a little less buggy that torch.eigh 
    # For this type of data

    # U: [n_samples, n_samples]
    # S: [min(n_samples, n_features)]
    # V: [n_features, n_features]
    U, S, V = torch.svd(Sigma)
    
    # Convert singular values to eigenvalues
    n_samples = data.shape[0]
    eigenvalues = (S ** 2) / (n_samples - 1)
    
    eigenvectors = V      
    
    #Step 3: If doing pca whitening we provide the option of returning a certain
    # num of principal components. 0 <= n_components < 1 indicates you want to keep
    # a certain percentage of explained variance. n_components > 1 indicates a 
    # you wish to keep that many. n_components = None means you want to keep all
    if algorithm == 'pca' and n_components is not None:

        if isinstance(n_components, float):
            if not 0 < n_components <= 1:
                raise ValueError("If n_components is float, it must be between 0 and 1")

            explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)
            cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)

            n_components = torch.sum(cumulative_variance_ratio <= n_components) + 1

        elif isinstance(n_components, int):
            if not 0 < n_components <= len(eigenvalues):
                raise ValueError(f"n_components must be between 1 and {len(eigenvalues)}")
        else:
            raise ValueError("n_components must be int or float")
                
        # Truncate eigenvalues and eigenvectors
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    return {
        'mean': mean,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        }

def apply_whitening_transform(
  data: torch.Tensor,
  stats: Dict,
  algorithm: str,
  epsilon: float = 1e-5
) -> torch.Tensor:
    """
    Apply whitening transform to data using pre-computed statistics.

    See https://arxiv.org/abs/1512.00809 for more details on transformations
    - Also gives two additional transforms that have not been implemented
    - Possible TODO

    See https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
    for details on PCA and ZCA in particular
    
    Args:
        data: Input data where unique data elements are along the first axis
        stats: Dict containing whitening statistics (mean, eigenvectors, eigenvalues)
        mode: Whitening mode, one of ['pca', 'zca', or 'cholesky]
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Whitened data of shape [N, D] for ZCA and cholesky or [N, D_reduced] for PCA
        where D_reduced is the number of components kept
    """
    data = data.reshape((-1, np.prod(data.size()[1:])))
    
    x_centered = data - stats.get('mean')
    
    # Calculate scaling matrix
    scaling = torch.diag(1. / torch.sqrt(stats.get('eigenvalues') + epsilon))
    
    if algorithm == 'pca':
        # For PCA: project onto eigenvectors and scale
        W = scaling @ stats.get('eigenvectors').T
    elif algorithm == 'zca':
        # For ZCA: project, scale, and rotate back
        W = (stats.get('eigenvectors') @
            scaling @ 
            stats.get('eigenvectors').T)
    elif algorithm == 'cholesky':
        # Based on Cholesky decomp, also related to QR decomp
        W = torch.linalg.cholesky(stats.get('eigenvectors') @
                                scaling @ 
                                stats.get('eigenvectors').T).T
    
    return x_centered @ W.T

def whitening_transform(
    images: torch.Tensor,
    algorithm = 'zca',
    stats = None,
    n_components = None,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Apply whitening to batched images.
    
    Args:
        images: Input images of shape [N, C, H, W]
        algorithm: which whitening algorithm to use
        stats: Pre-computed statistics for whitening, computes for the given 
               batch if none provided
        n_components: Used for PCA algorithm
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Whitened images of shape [N, C, H, W]
    """

    N, C, H, W = images.shape

    if stats is None:
      stats = compute_whitening_stats(images,
                                      algorithm = algorithm,
                                      n_components = n_components)
      print('none')

    else:
      if ('mean' not in stats or 
          'eigenvalues' not in stats or 
          'eigenvectors' not in stats
          ):
        raise ValueError("stats must contain mean, eigenvalues, and eigenvectors")
    
    whitened_flat = apply_whitening_transform(images, stats, algorithm, epsilon)


    # Because of the way you truncate the eigenvalues/vectors
    # You end up with a matrix that you cannot reshape into image format in a 
    # general manner
    if algorithm == 'pca' and n_components is not None:
        return whitened_flat
    else:
        # For other methods, return in original shape
        return whitened_flat.reshape(N, C, H, W)
    
def whiten(images: torch.Tensor, algorithm: str,  stats = None, **kwargs) -> torch.Tensor:
  """
    Wrapper for all whitening transformations

    Args:
        images: tensor of shape (N, C, H, W)
        algorithm: what whitening transform we want to use
        stats: dictionary of dataset statistics needed for whitening transformations
  """

  if algorithm == 'frequency':
    return frequency_whitening(images, **kwargs)
  elif algorithm == 'pca' or algorithm == 'zca' or algorithm == 'cholesky':
    return whitening_transform(images, algorithm, stats, **kwargs)
  else:
    raise ValueError(f"Unknown whitening algorithm: {algorithm}, must be one of ['frequency', 'pca', 'zca', 'cholesky]")
  
class WhiteningTransform(object):
    """
    A PyTorch transform for image whitening that can be used in a transform pipeline.
    Supports frequency, PCA, and ZCA whitening methods.
    """
    def __init__(
        self,
        algorithm: str = 'zca',
        stats: Optional[Dict] = None,
        compute_stats: bool = False,
        **kwargs
    ):
        """
        Initialize whitening transform.
        
        Args:
            algorithm: One of ['frequency', 'pca', 'zca']
            stats: Pre-computed statistics for PCA/ZCA whitening
            compute_stats: If True, will compute stats on first batch seen
            **kwargs: Additional arguments passed to whitening function
        """
        self.algorithm = algorithm
        self.stats = stats
        self.compute_stats = compute_stats
        self.kwargs = kwargs
    

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply whitening transform to images.
        
        Args:
            images: Input images of shape [N, C, H, W] or [C, H, W]
        
        Returns:
            Whitened images of same shape as input
        """
        # Add batch dimension if necessary
        if images.dim() == 3:
            images = images.unsqueeze(0)
            single_image = True
        else:
            single_image = False

        # Apply whitening
        whitened = whiten(
            images,
            self.algorithm,
            self.stats,
            **self.kwargs
        )
     
        # Remove batch dimension if input was single image
        if single_image:
            whitened = whitened.squeeze(0)
            
        return whitened

    def __repr__(self):
        return "custom augmentation"
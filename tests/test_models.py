"""Architecture checks for the Task 1 autoencoder."""
import torch

from src.models.autoencoder import ConvAutoencoder


def test_shapes_and_range():
    m = ConvAutoencoder(base_channels=24, bottleneck_dim=512).eval()
    x = torch.rand(2, 3, 128, 128)
    z = m.encode(x)
    assert z.shape == (2, 8, 8, 8) and z[0].numel() == 512        # latent size = bottleneck_dim
    y = m(x)
    assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1


def test_output_depends_only_on_the_latent():
    """No skip connections: the decoder sees nothing but the latent code.
    So forward(x) == decode(encode(x)), and two different inputs with the same latent give the same output."""
    m = ConvAutoencoder(base_channels=24, bottleneck_dim=512).eval()
    x1, x2 = torch.rand(1, 3, 128, 128), torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        assert torch.allclose(m(x1), m.decode(m.encode(x1)))
        z = m.encode(x1)
        assert torch.allclose(m.decode(z), m.decode(z.clone()))
        assert (m(x1) - m(x2)).abs().max() > 0          # different inputs -> different outputs
    compression = (3 * 128 * 128) / z.numel()
    assert compression == 96

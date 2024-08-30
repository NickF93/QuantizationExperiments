import unittest
import torch
import logging
from src.quantizer import linear_q_with_scale_and_zero_point, linear_dequantizer
from src.helper import plot_quantization_errors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestQuantizer(unittest.TestCase):

    def setUp(self):
        torch.seed(123)

        ### a dummy tensor to test the implementation
        self.test_tensor = torch.tensor(
            [[191.60, -13.50,  728.60],
             [ 92.14, 295.50, -184.00],
             [  0.00, 684.60,  245.50]]
        )
        
        ### a dummy tensor to test the implementation
        self.quantized_tensor = torch.tensor(
            [[-15,  -74,  127], 
             [-44,   14, -123], 
             [-70,  126,    0]],
             dtype=torch.int8)

    def test_quantizer(self):

        ### these are random values for "scale" and "zero_point"
        ### to test the implementation
        scale = 3.5
        zero_point = -70

        quantized_tensor = linear_q_with_scale_and_zero_point(self.test_tensor, scale, zero_point)

        # Log the original tensor
        logger.info('test_tensor:\n%s', self.test_tensor.numpy())
        logger.info('quantized_tensor:\n%s', quantized_tensor.numpy())

        logger.info('tensor:\n%s', self.quantized_tensor.numpy())

        self.assertTrue(torch.equal(self.quantized_tensor, quantized_tensor))

    def test_dequantizer(self):

        ### these are random values for "scale" and "zero_point"
        ### to test the implementation
        scale = 3.5
        zero_point = -70

        dequantized_tensor = linear_dequantizer(self.quantized_tensor, scale, zero_point)

        # Log the quantized and dequantized tensors
        logger.info('quantized_tensor:\n%s', self.quantized_tensor.numpy())
        logger.info('dequantized_tensor:\n%s', dequantized_tensor.numpy())

        tensor = torch.as_tensor([[ 192.5, - 14.0,  689.5],
                                  [  91.0,  294.0, -185.5],
                                  [   0.0,  686.0,  245.0]])
        logger.info('tensor:\n%s', tensor.numpy())

        plot_quantization_errors(self.test_tensor, self.quantized_tensor, dequantized_tensor)

        # Use torch.allclose for floating-point comparisons
        self.assertTrue(torch.allclose(tensor, dequantized_tensor, atol=1e-9))

if __name__ == '__main__':
    unittest.main()

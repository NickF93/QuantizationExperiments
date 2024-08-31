import unittest
import torch
import numpy as np
import logging
from src.quantizer import linear_q_with_scale_and_zero_point, linear_dequantizer, get_q_scale_and_zero_point, linear_quantizer
from src.helper import plot_quantization_errors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestQuantizer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(123)

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
        logger.warning('test_quantizer')

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
        logger.warning('test_dequantizer')

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
    
    def test_newscale(self):
        logger.warning('test_newscale')

        new_scale, new_zero_point = get_q_scale_and_zero_point(self.test_tensor)

        scale = 3.58
        zero_point = -77

        logger.info('new_scale: %s', new_scale)
        logger.info('new_zero_point: %s', new_zero_point)

        logger.info('SCALE ERR: %s', np.abs(scale - new_scale))
        logger.info('ZEROP ERR: %s', np.abs(zero_point - new_zero_point))

        self.assertTrue(
            np.allclose(new_scale, scale, atol=1e-2) and
            np.allclose(new_zero_point, zero_point, atol=1e-2)
        )

    def test_scaled_quantization_and_dequantization(self):
        logger.warning('test_scaled_quantization_and_dequantization')

        r_tensor = torch.randn((4, 4))

        quantized_tensor, new_scale, new_zero_point = linear_quantizer(r_tensor)
        dequantized_tensor = linear_dequantizer(quantized_tensor, new_scale, new_zero_point)

        plot_quantization_errors(r_tensor, quantized_tensor, dequantized_tensor)

        error = (dequantized_tensor-r_tensor).square().mean()

        logger.info('error: %s', error.numpy())

        self.assertTrue(
            np.allclose(error, 0.0, atol=1e-4)
        )

if __name__ == '__main__':
    unittest.main()

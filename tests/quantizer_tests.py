import unittest
import torch
import logging
from src.quantizer import linear_q_with_scale_and_zero_point, linear_dequantizer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestMyModule(unittest.TestCase):

    def test_quantizer(self):
        ### a dummy tensor to test the implementation
        test_tensor = torch.tensor(
            [[191.6, -13.5, 728.6],
             [92.14, 295.5, -184],
             [0, 684.6, 245.5]]
        )

        ### these are random values for "scale" and "zero_point"
        ### to test the implementation
        scale = 3.5
        zero_point = -70

        quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale, zero_point)

        # Log the original tensor
        logger.debug('test_tensor:\n%s', test_tensor.numpy())
        logger.debug('quantized_tensor:\n%s', quantized_tensor.numpy())

        tensor = torch.as_tensor([[-15, -74, 127], 
                                  [-44, 14, -123], 
                                  [-70, 126, 0]], dtype=torch.int8)
        logger.debug('tensor:\n%s', tensor.numpy())

        self.assertTrue(torch.equal(tensor, quantized_tensor))

    def test_dequantizer(self):
        ### a dummy tensor to test the implementation

        quantized_tensor = torch.as_tensor([[- 15, - 74,  127], 
                                            [- 44,   14, -123], 
                                            [- 70,  126,    0]], dtype=torch.int8)

        ### these are random values for "scale" and "zero_point"
        ### to test the implementation
        scale = 3.5
        zero_point = -70

        dequantized_tensor = linear_dequantizer(quantized_tensor, scale, zero_point)

        # Log the quantized and dequantized tensors
        logger.debug('quantized_tensor:\n%s', quantized_tensor.numpy())
        logger.debug('dequantized_tensor:\n%s', dequantized_tensor.numpy())

        tensor = torch.as_tensor([[ 192.5, - 14.0,  689.5],
                                  [  91.0,  294.0, -185.5],
                                  [   0.0,  686.0,  245.0]])
        logger.debug('tensor:\n%s', tensor.numpy())

        # Use torch.allclose for floating-point comparisons
        self.assertTrue(torch.allclose(tensor, dequantized_tensor, atol=1e-9))

if __name__ == '__main__':
    unittest.main()

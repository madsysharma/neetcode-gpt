import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = np.max(z)
        new_arr = np.subtract(z, max_val)
        exp = np.exp(new_arr)
        tot = np.sum(exp)
        result = np.divide(exp, tot)
        return np.round(result, 4)

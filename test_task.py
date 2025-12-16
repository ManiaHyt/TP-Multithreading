import unittest
import numpy as np
import numpy.testing as npt
from task import Task


class TestTask(unittest.TestCase):
    def test_task_solves_ax_equals_b(self):
        np.random.seed(0)
        task = Task(identifier=1, size=10)
        task.work()
        npt.assert_allclose(task.a @ task.x, task.b, rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    unittest.main()

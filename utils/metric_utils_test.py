import unittest

from utils.metric_utils import numpy_metrics_at_thresholds
import numpy as np

class MetricTest(unittest.TestCase):
    def test_numpy_metrics_at_thresholds(self):
        trues = [[0, 1], [1, 1]]
        probs = [[0.2, 0.5], [0.3, 0.9]]
        accuracy, precisions, recalls, f1scores, thresholds = numpy_metrics_at_thresholds(trues, probs, threshold_num=2)
        self.assertTrue(equal(thresholds, [-1e-9, 1 + 1e-9]))
        self.assertTrue(equal(accuracy, [0.5, 0]))
        self.assertTrue(equal(precisions, [0.75, 0]))
        self.assertTrue(equal(recalls, [1, 0]))
        self.assertTrue(equal(f1scores, [5/6, 0]))
        accuracy, precisions, recalls, f1scores, thresholds = numpy_metrics_at_thresholds(trues[0], probs[0], threshold_num=2)
        self.assertTrue(equal(thresholds, [-1e-9, 1 + 1e-9]))
        self.assertTrue(equal(accuracy, [0,0]))
        self.assertTrue(equal(precisions, [0.5,0]))
        self.assertTrue(equal(recalls, [1,0]))
        self.assertTrue(equal(f1scores, [2/3,0]))

def equal(list1,list2):
    list1=np.array(list1)
    list2=np.array(list2)
    if list1.shape!=list2.shape:
        return False
    sub=np.abs(list1-list2)
    return np.all(sub<1e-6)
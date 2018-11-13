import numpy as np
from mxnet import nd
import pdb


def _to_numpy(*args):
    ret = []
    for arr in args:
        if isinstance(arr, nd.NDArray):
            ret.append(arr.asnumpy())
        elif isinstance(arr, np.ndarray):
            ret.append(arr)
        else:
            raise TypeError('Expect type: (numpy.ndarrary, mxnet.nd.NDArray) '
                            'but get {}'.format(type(arr)))

    if len(ret) == 1:
        ret = ret[0]
    return ret


class Metric(object):
    def __init__(self, label_names, name='metric'):
        self.label_names = label_names
        self.metric_name = name

    def get(self):
        raise NotImplementedError

    def get_name_value(self):
        return dict(zip(*self.get()))

    def get_average(self):
        return float(np.mean(self.get()[1]))

    def update(self, labels, preds):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def _check_shape(self, labels, preds):
        label_shape, pred_shape = labels.shape, preds.shape
        if label_shape != pred_shape:
            raise ValueError("Shape of labels {} does not match shape of "
                             "predictions {}".format(label_shape, pred_shape))
        if label_shape[1] != len(self.label_names):
            raise ValueError("Dimension {} does not match num of "
                             "label_names {}".format(label_shape[1], len(self.label_names)))


class Accuracy(Metric):
    def __init__(self, label_names, name='acc'):
        super(Accuracy, self).__init__(label_names, name)
        self.num_correct = np.zeros(len(self.label_names), dtype=np.int32)
        self.num_total = np.zeros(len(self.label_names), dtype=np.int32)

    def get(self):
        return self.label_names, np.divide(self.num_correct, self.num_total).tolist()

    def update(self, labels, preds):
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        assert len(labels) == len(preds)
        for label, pred in zip(labels, preds):
            self._check_shape(label, pred)
            label, pred = _to_numpy(label, pred)
            valid = np.not_equal(label, 999)
            correct = np.equal(pred, label)
            self.num_correct += np.sum(np.bitwise_and(valid, correct).astype(np.int32), axis=0)
            # self.num_total += np.sum(valid.astype(np.int32), axis=0)
            self.num_total += (np.ones(label.shape[1], dtype=np.int32) * label.shape[0])

    def reset(self):
        self.num_correct[:] = 0
        self.num_total[:] = 0


class F1(Metric):
    def __init__(self, label_names, name='f1'):
        super(F1, self).__init__(label_names, name)
        self.true_positive = np.zeros(len(self.label_names), dtype=np.int32)
        self.true_negative = np.zeros(len(self.label_names), dtype=np.int32)
        self.false_positive = np.zeros(len(self.label_names), dtype=np.int32)
        self.false_negative = np.zeros(len(self.label_names), dtype=np.int32)
        self.num_total = np.zeros(len(self.label_names), dtype=np.int32)

    def get(self):
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return self.label_names, np.where(denominator > 0, 2 * self.true_positive / denominator, 0).tolist()

    def update(self, labels, preds):
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        assert len(labels) == len(preds)
        for label, pred in zip(labels, preds):
            self._check_shape(label, pred)
            label, pred = _to_numpy(label, pred)

            self.num_total += np.sum(np.not_equal(label, 999).astype(np.int32), axis=0)
            self.true_positive += np.sum(np.bitwise_and(np.equal(label, 1), np.equal(pred, 1)).astype(np.int32), axis=0)
            self.true_negative += np.sum(np.bitwise_and(np.equal(label, 0), np.equal(pred, 0)).astype(np.int32), axis=0)
            self.false_positive += np.sum(np.bitwise_and(np.equal(label, 0), np.equal(pred, 1)).astype(np.int32), axis=0)
            self.false_negative += np.sum(np.bitwise_and(np.equal(label, 1), np.equal(pred, 0)).astype(np.int32), axis=0)

    def reset(self):
        self.true_positive[:] = 0
        self.true_negative[:] = 0
        self.false_positive[:] = 0
        self.false_negative[:] = 0
        self.num_total[:] = 0


class CorruptionMatrix(Metric):
    def __init__(self, label_names, name='corruption'):
        super(CorruptionMatrix, self).__init__(label_names, name)
        self.C_accum = np.zeros((len(label_names), 2, 2))
        self.num_example = np.zeros((len(label_names), 2))

    def update(self, labels, probs):
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        if not isinstance(probs, (list, tuple)):
            probs = [probs]
        assert len(labels) == len(probs)
        for label, prob in zip(labels, probs):
            self._check_shape(label, prob)
            label, prob = _to_numpy(label, prob)
            self.C_accum[:, 0, 0] += np.sum(np.bitwise_and(label == 1, prob >= 0.5) * prob, axis=0)
            self.C_accum[:, 0, 1] += np.sum(np.bitwise_and(label == 1, prob < 0.5) * (1 - prob), axis=0)
            self.C_accum[:, 1, 0] += np.sum(np.bitwise_and(label == 0, prob >= 0.5) * prob, axis=0)
            self.C_accum[:, 1, 1] += np.sum(np.bitwise_and(label == 0, prob < 0.5) * (1 - prob), axis=0)

            self.num_example[:, 0] += np.sum(label == 1, axis=0)
            self.num_example[:, 1] += np.sum(label == 0, axis=0)

    def get(self):
        C = self.C_accum / np.repeat(np.expand_dims(self.num_example, 2), 2, axis=2)
        return self.label_names, C.reshape((-1, 4)).tolist()

    def reset(self):
        self.C_accum[:] = 0
        self.num_example[:] = 0


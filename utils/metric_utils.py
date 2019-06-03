from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import tensorflow as tf


def sequence_binary_tagging_metric_fn(per_example_loss, label_ids, logits, input_mask, is_real_example):
    predictions = tf.where(tf.logical_and(logits >= 0, input_mask == 1), tf.ones(tf.shape(logits)),
                           tf.zeros(tf.shape(logits)))
    accuracy = tf.metrics.accuracy(
        labels=label_ids, predictions=predictions, weights=is_real_example)

    loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
    return {
        "eval_accuracy": accuracy,
        "eval_loss": loss,
    }


def sequence_tagging_metric_fn(per_example_loss, label_ids, logits, input_mask, is_real_example):
    """In default, use num_label-1 as mask id."""
    num_label=logits.shape[-1].value
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    shape = tf.shape(predictions)
    predictions = tf.where(input_mask == 1, predictions, tf.fill(shape, num_label - 1))
    weights=tf.where(label_ids==num_label-1 ,tf.zeros(shape),tf.ones(shape))
    accuracy = tf.metrics.accuracy(
        labels=label_ids, predictions=predictions, weights=weights)

    loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
    return {
        "eval_accuracy": accuracy,
        "eval_loss": loss,
    }


def single_label_classification_metric_fn(per_example_loss, label_ids, logits, input_mask, is_real_example):
    predictions = tf.argmax(logits, axis=-1)
    accuracy = tf.metrics.accuracy(
        labels=label_ids, predictions=predictions, weights=is_real_example)

    loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
    return {
        "eval_accuracy": accuracy,
        "eval_loss": loss,
    }


def multi_label_tagging_metric_fn(per_example_loss, label_ids, logits, input_mask, is_real_example):
    return sequence_binary_tagging_metric_fn(per_example_loss, label_ids, logits, input_mask, is_real_example)


def report_metrics(trues, preds, labels_list=None):
    trues=np.array(trues)
    preds=np.array(preds)
    assert trues.shape==preds.shape
    if trues.ndim==1:
        precision, recall, fscore, _=precision_recall_fscore_support(trues,preds,average='micro')
    else:
        metric_results=[precision_recall_fscore_support(true, pred, labels=labels_list, average='micro')
                                                        for pred, true in zip(preds, trues)]

        precision, recall, fscore, _ = list(zip(*metric_results))
        precision, recall, fscore=np.apply_along_axis(np.mean,1,[precision, recall, fscore])

    accuracy=get_accuracy(trues,preds)

    report = 'accuracy: {}\n'.format(accuracy) + \
             'precision: {}\n'.format(precision) + \
             'recall: {}\n'.format(recall) + \
             'fscore: {}\n'.format(fscore)
    return report

def threshold_selection(y_true, probs, threshold_num=20,f1_weight=0.5):
    assert threshold_num>1
    accuracy, precisions, recalls, f1scores,thresholds=numpy_metrics_at_thresholds(y_true, probs,
                                                                        threshold_num=threshold_num)
    score=f1_weight*f1scores+accuracy*(1-f1_weight)
    best=thresholds[np.argmax(score)]
    tf.logging.info("best threshold is {}".format(best))
    return best


def numpy_metrics_at_thresholds(y_true, probs,  threshold_num=20, threshold=0.5):
    epsilon = 1e-9
    probs = np.array(probs)
    y_true = np.array(y_true)
    assert y_true.shape==probs.shape

    if probs.ndim==1:
        probs=np.expand_dims(probs,0)
        y_true=np.expand_dims(y_true,0)

    y_true = y_true == 1
    total_true = np.sum(y_true, axis=-1)

    if threshold_num:
        assert threshold_num > 1
        thresholds = [0 - epsilon] + [i / (threshold_num - 1) for i in range(1, threshold_num - 1)] + [1 + epsilon]
    elif threshold is not None:
        thresholds = [threshold]
    else:
        raise ValueError('Either threshold_num or threshold must be provided.')

    true = np.array([probs >= threshold for threshold in thresholds])

    accuracy = get_accuracy(y_true,true)
    tps = np.sum(y_true & true, -1)
    precisions = tps / (np.sum(true, -1) + epsilon)
    recalls = tps / total_true
    f1scores = 2 * precisions * recalls / (precisions + recalls + epsilon)

    precisions = np.mean(precisions, -1)
    recalls = np.mean(recalls, -1)
    f1scores = np.mean(f1scores, -1)

    return accuracy, precisions, recalls, f1scores,thresholds


def metrics_at_thresholds(y_true, probs, threshold_num=20,threshold=0.5):
    epsilon = 1e-9
    total_true = tf.reduce_sum(y_true, axis=-1)
    y_true = tf.equal(y_true, 1)

    if threshold_num:
        assert threshold_num > 1
        thresholds = [0 - epsilon] + [i / (threshold_num - 1) for i in range(1, threshold_num - 1)] + [1 + epsilon]
    elif threshold is not None:
        thresholds = [threshold]
    else:
        raise ValueError('Either threshold_num or threshold must be provided.')

    true = tf.stack([probs >= threshold for threshold in thresholds])

    accuracy = tf.reduce_all(tf.equal(true, y_true), axis=-1)
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32), axis=-1)

    tps = tf.cast(tf.logical_and(y_true, true), tf.float32)
    tps = tf.reduce_sum(tps, axis=-1)
    precisions = tps / tf.add(tf.reduce_sum(tf.cast(true, tf.float32), -1), epsilon)
    recalls = tps / tf.cast(total_true, tf.float32)
    f1scores = 2 * precisions * recalls / (precisions + recalls + epsilon)

    precisions=tf.reduce_mean(precisions,-1)
    recalls=tf.reduce_mean(recalls,-1)
    f1scores=tf.reduce_mean(f1scores,-1)
    return accuracy, precisions, recalls, f1scores, thresholds

def get_accuracy(y_true,y_pred):
    accuracy = np.sum(np.all(y_pred == y_true, axis=-1), -1) / len(y_true)
    return accuracy
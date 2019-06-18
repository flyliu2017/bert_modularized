import os

from processor.data_processor import MultiLabelClassificationProcessor, InputExample, threshold_selection
import numpy as np
import tensorflow as tf

class IntentClassificationProcessor(MultiLabelClassificationProcessor):
    def get_train_examples(self):
        """See base class."""
        path=os.path.join(self.data_dir,'train')
        return self._create_examples(path)

    def get_dev_examples(self):
        """See base class."""
        path = os.path.join(self.data_dir, 'eval')
        return self._create_examples(path)

    def get_test_examples(self):
        """See base class."""
        path = os.path.join(self.data_dir, 'test')
        return self._create_examples(path)

    def get_prediction_examples(self,input_file):
        return self._create_examples(input_file)


    def get_labels(self):
        with open(os.path.join(self.data_dir,'intent_set'), 'r', encoding='utf8') as f:
            intent_set = f.read().splitlines()
        return intent_set

    def _create_examples(self, file):
        examples = []
        DELIMITER=' ||| '
        set_type=os.path.basename(file)
        with open(file, 'r', encoding='utf8') as f:
            datas = f.read().splitlines()

        for (i, n) in enumerate(datas):
            content,intents,seq = n.split(DELIMITER)
            guid = "%s-%s" % (set_type, i)
            intents = intents.split('\t')
            examples.append(
                InputExample(guid=guid, text_a=content, text_b=seq, label=intents))

        return examples

    def post_process(self, output_dir, label_ids, probabilities, input_mask, input_ids,threshold):
        best_threshold,best_metrics=threshold_selection(label_ids,probabilities)
        tf.logging.info('best_threshold:{}'.format(best_threshold))
        super().post_process(output_dir, label_ids, probabilities, input_mask, input_ids,best_threshold)
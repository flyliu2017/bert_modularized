import os

from processor.data_processor import InputExample, SequenceTaggingProcessor
from model.model_fn import create_sequence_binary_tagging_model


class ExtractPhrasesProcessor(SequenceTaggingProcessor):

    def get_train_examples(self):
        """See base class."""
        return self._create_examples("train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples("eval")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples("test")

    def get_labels(self):
        return ['0', '1']

    @property
    def create_model(self):
        return create_sequence_binary_tagging_model

    def _create_examples(self, set_type):
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            labels = f.read().splitlines()

        for (i, n) in enumerate(zip(txts, labels)):
            txt, label = n
            guid = "%s-%s" % (set_type, i)
            text_a, text_b = txt.split(' | ')
            label = label.split(' | ')[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def create_label_features(self, example, tokens):
        label_ids=[0]*self.max_seq_length
        for label in example.label:
            label_ids[self.label_map[label]]=1
        return label_ids

class ExtractPhrasesFromSegmentedInputProcessor(ExtractPhrasesProcessor):
    """This processor is for tagging characters rather than tokens"""

    def _create_examples(self, set_type):
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            labels = f.read().splitlines()

        for (i, n) in enumerate(zip(txts, labels)):
            txt, label = n
            guid = "%s-%s" % (set_type, i)
            text_a, text_b = txt.split(' | ')
            text_a = ' '.join(list(text_a))
            label = label.split(' | ')[0]
            label = ' '.join(list(label))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

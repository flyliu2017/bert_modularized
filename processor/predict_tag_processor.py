import os

from processor.data_processor import SingleLabelClassificationProcessor, InputExample, MultiLabelClassificationProcessor


class SingleTagPredictionProcessor(SingleLabelClassificationProcessor):
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
        ret = []
        for line in self._read_tsv("/data/share/liuchang/comments_dayu/tag_prediction/data/tag_vocab.txt"):
            ret.append(line[0])
        return ret

    def _create_examples(self, set_type):
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_converted_tag.txt'.format(set_type)), 'r', encoding='utf8') as f:
            phrases = f.read().splitlines()

        for (i, n) in enumerate(zip(txts, phrases)):
            txt, phrase = n
            guid = "%s-%s" % (set_type, i)
            tag = txt.split(' | ')[-1]
            phrase = phrase.split(' | ')[0]
            examples.append(
                InputExample(guid=guid, text_a=phrase, text_b=None, label=tag))

        return examples


class MultiTagPredictionProcessor(MultiLabelClassificationProcessor):
    def get_train_examples(self):
        """See base class."""
        return self._create_examples("train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples("eval")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples("test")

    def get_prediction_examples(self,input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            phrases = f.read().splitlines()
        examples=[ InputExample(guid=str(i), text_a=phrase, text_b=None, label=[])
                    for i,phrase in enumerate(phrases)]

        return examples


    def get_labels(self):
        ret = []
        for line in self._read_tsv("/data/share/liuchang/comments_dayu/tag_prediction/data/tag_vocab.txt"):
            ret.append(line[0])
        return ret

    def _create_examples(self, set_type):
        examples = []
        with open(os.path.join(self.data_dir, '{}_phrases_tags'.format(set_type)), 'r', encoding='utf8') as f:
            tags = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_phrases'.format(set_type)), 'r', encoding='utf8') as f:
            phrases = f.read().splitlines()

        for (i, n) in enumerate(zip(tags, phrases)):
            tag, phrase = n
            guid = "%s-%s" % (set_type, i)
            tag = tag.split(' | ')
            examples.append(
                InputExample(guid=guid, text_a=phrase, text_b=None, label=tag))

        return examples

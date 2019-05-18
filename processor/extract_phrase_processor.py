import os

from processor.data_processor import InputExample, SequenceTaggingProcessor,SequenceBinaryTaggingProcessor, _truncate_seq_pair, PaddingInputExample
from model.model_fn import create_sequence_binary_tagging_model, create_sequence_tagging_model


class ExtractPhrasesProcessor(SequenceBinaryTaggingProcessor):

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

class ExtractAllPhrasesProcessor(ExtractPhrasesProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def _create_examples(self, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_multitags'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_multitags'.format(set_type)), 'r', encoding='utf8') as f:
            labels = f.read().splitlines()


        for (i, n) in enumerate(zip(txts, labels)):
            txt, label = n
            guid = "%s-%s" % (set_type, i)
            text_a=txt.split(' | ')[0]
            text_a = ' '.join(list(text_a))
            label = label.split(' | ')
            label = [' '.join(list(l)) for l in label]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples

    def create_label_features(self, example, tokens):
        label_list = example.label
        if not isinstance(label_list, list):
            label_list = [label_list]
        label_tokens = [self.tokenizer.tokenize(label) for label in label_list]

        label_id = [0] * self.max_seq_length
        for label in label_tokens:
            length = len(label)
            for i in range(len(tokens) - length + 1):
                if tokens[i:i + length] == label:
                    start = i
                    end = i + length
                    label_id[start:end] = [1] * (end - start)
                    break

            else:
                raise ValueError("can't find phrase in text.")

class ExtractAllPhrasesAndTagsProcessor(SequenceTaggingProcessor):
    """Processor for the MRPC data set (GLUE version)."""

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
        with open('/data/share/liuchang/comments_dayu/tag_prediction/data/tag_vocab.txt', 'r', encoding='utf8') as f:
            tags = f.read().splitlines()
        return tags+['None_tag']

    def _create_examples(self, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(os.path.join(self.data_dir, '{}_xs_multitags'.format(set_type)), 'r', encoding='utf8') as f:
            comments = f.read().splitlines()
        with open(os.path.join(self.data_dir, '{}_ys_multitags'.format(set_type)), 'r', encoding='utf8') as f:
            phrases = f.read().splitlines()


        for (i, n) in enumerate(zip(comments, phrases)):
            comment, phrase = n
            guid = "%s-%s" % (set_type, i)
            txt_split = comment.split(' | ')
            text_a= txt_split[0]
            text_a = ' '.join(list(text_a))

            tags=txt_split[1:]
            phrase_list = phrase.split(' | ')
            phrase_list = [' '.join(list(n)) for n in phrase_list]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=phrase_list, label=tags))

        return examples

    def create_input_features(self, example):
        if isinstance(example, PaddingInputExample):
            return self.padding_input_features

        tokens_a = self.tokenizer.tokenize(example.text_a)


        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_seq_length:
            padding_length=self.max_seq_length-len(input_ids)
            input_ids.extend([0]*padding_length)
            input_mask.extend([0]*padding_length)
            segment_ids.extend([0]*padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        return tokens, input_ids, input_mask, segment_ids

    def create_label_features(self, example, tokens):
        phrase_list = [self.tokenizer.tokenize(phrase) for phrase in example.text_b]
        tags=example.label
        tag_ids=[self.label_map[tag] for tag in tags]

        label_id = [len(self.label_list)-1] * self.max_seq_length
        for phrase,tag_id in zip(phrase_list,tag_ids):
            length = len(phrase)
            for i in range(len(tokens) - length + 1):
                if tokens[i:i + length] == phrase:
                    start = i
                    end = i + length
                    label_id[start:end] = [tag_id] * (end - start)
                    break

            else:
                raise ValueError("can't find phrase in text.")

        return label_id
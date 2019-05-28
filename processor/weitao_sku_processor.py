import os
from processor.predict_tag_processor import SingleTagPredictionProcessor,InputExample

class AlimamaTitleClassificationPorcessor(SingleTagPredictionProcessor):
    def get_labels(self):
        with open(os.path.join(self.data_dir,'categories_1769'), 'r', encoding='utf8') as f:
            labels = f.read().splitlines()
        return labels

    def _create_examples(self, set_type):
        examples = []
        with open(os.path.join(self.data_dir, '{}_alimama_sku'.format(set_type)), 'r', encoding='utf8') as f:
            txts = f.read().splitlines()

        for (i, txt) in enumerate(txts):
            guid = "%s-%s" % (set_type, i)
            label,title = txt.split(' ',1)
            label = label.split('__',2)[-1]
            examples.append(
                InputExample(guid=guid, text_a=title, text_b=None, label=label))

        return examples




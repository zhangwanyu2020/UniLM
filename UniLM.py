from __future__ import print_function
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 获取数据
def get_data(data_path):
    lines = []
    with open(data_path,encoding='UTF-8-sig') as f:
        data = f.readlines()
        for line in data:
            lines.append(line.strip().replace(' ',''))
    return lines
train_x_path = '/content/drive/My Drive/test_data/train_x.txt'
train_y_path = '/content/drive/My Drive/test_data/train_y.txt'
train_x = get_data(train_x_path)
train_y = get_data(train_y_path)
assert len(train_x) == len(train_y)
train_data_all = []
for x,y in zip(train_x,train_y):
    train_data_all.append((x,y))

train_data = train_data_all[0:int(len(train_data_all)*0.99)]
valid_data = train_data_all[int(len(train_data_all)*0.9995):]
print(valid_data[0])
print('train_data & valid_data have builded .......')
# 基本参数
maxlen = 400
batch_size = 8
epochs = 5

# bert配置
config_path = '/content/drive/My Drive/publish/bert_config.json'
checkpoint_path = '/content/drive/My Drive/publish/bert_model.ckpt'
dict_path = '/content/drive/My Drive/publish/vocab.txt'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (content, title) in self.sample(random):
            # if total_length <= maxlen:
            #     break
            # elif len(first_sequence) > len(second_sequence):
            #     first_sequence.pop(pop_index)
            # else:
            #     second_sequence.pop(pop_index)
            token_ids, segment_ids = tokenizer.encode(content, title, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

loss = CrossEntropy(2)(model.inputs + model.outputs)
print(loss)

model = Model(model.inputs, loss)
model.compile(optimizer=Adam(1e-5))
# model.summary()

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=40)

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights('/content/sample_data/best_model.weights')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)

    def evaluate(self, data, topk=2):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for content, title in tqdm(data):
            total += 1
            title = ' '.join(title)
            pred_title = ' '.join(autotitle.generate(content, topk))
            print('real_title:',title)
            print('pred_title:',pred_title)
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[title.split(' ')],
                    hypothesis=pred_title.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }

if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    # model.load_weights('/content/sample_data/best_model.weights')
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )


# model.load_weights('./best_model.weights')
# evaluator = Evaluator()
# valid_generator = data_generator(valid_data, batch_size)
# total = 0
# topk = 2
# for title, content in tqdm(valid_data):
#     total += 1
#     title = ' '.join(title)
#     pred_title = ' '.join(autotitle.generate(content, topk))
#     print(title)
#     print('***',pred_title)
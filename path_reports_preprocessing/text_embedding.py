import numpy as np
import pickle
import torch
from operator import itemgetter
from pytorch_transformers import *

# PyTorch-Transformers has a unified API
# for 6 transformer architectures and 27 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = {'BERT': (BertModel, BertTokenizer, 'bert-base-uncased'),
          'GPT': (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
          'GPT2': (GPT2Model, GPT2Tokenizer, 'gpt2'),
          'XL': (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
          'XLNet': (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
          'XLM': (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024')}


if __name__ == '__main__':

    model_class, tokenizer_class, pretrained_weights = MODELS['BERT']

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Encode text
    with open('path_reports_preprocessing/list_of_reports.pkl', 'rb') as fh:
        data = pickle.load(fh)

    window_size = 10
    for i, report in enumerate(data):
        tokens = report.split(' ')
        print(i)
        while len(tokens) > 0:
            idx = np.arange(0, int(np.minimum(window_size, len(tokens))))
            sentence = " ".join(itemgetter(*idx)(tokens))
            print(sentence)
            input_ids = torch.tensor([tokenizer.encode(sentence)])
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
                print(last_hidden_states.shape)

            for i in sorted(idx, reverse=True):
                del tokens[i]

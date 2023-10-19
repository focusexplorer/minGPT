import unittest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from transformers import LineByLineTextDataset
from mingpt.trainer import Trainer

from transformers import pipeline, set_seed
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import GPT2TokenizerFast

import sys


def train_tokenizer(files):
    SAVE_PATH = "./token"
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    trainer = BpeTrainer(vocab_size=50000, show_progress=True,
                         inital_alphabet=ByteLevel.alphabet(), special_tokens=special_tokens)
    tokenizer.train(files, trainer)
    newtokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    newtokenizer.save_pretrained(SAVE_PATH)
    # load tokenizer from pretrained
    tokenizer = GPT2Tokenizer.from_pretrained(SAVE_PATH)
    tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>"})
    return tokenizer

def create_dataset(tokenizer):
    # save dir
    # setting train data
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="./text/sanguoyanyi.txt",
        block_size=32,
    )
    return dataset
# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
if __name__ == '__main__':
    files = ["text/remeo_and_juliet.txt"]
    tokenizer = train_tokenizer(files)
    sys.exit(0)
    model_type = 'gpt2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = create_dataset()
    model=GPT()

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # many possible options, see the file
    train_config.max_iters = 1000
    train_config.batch_size = 32
    trainer = Trainer(train_config, model, train_dataset)
    trainer.run()
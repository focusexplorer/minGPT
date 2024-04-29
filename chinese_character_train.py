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
import re

from mingpt.trainer import Trainer

block_size=10
def cut_pad(m):
    cc=block_size
    if cc==0:
        raise Exception("block_size=0")
    if m.size()[0] > cc:
        m2 = m[0:cc]
    else:
        m2 = torch.nn.functional.pad(m, (cc-m.size()[0],0), 'constant', 0)
    return m2;
class Data:
    def __init__(self, file, char2index,index2char):
        self.data=[]
        self.raw_data=[]
        with open(file, 'r') as f:
            for line in f:
                for char in line:
                    self.raw_data.append(char2index[char])

    def __len__(self):
        return len(self.raw_data)-block_size-1

    def __getitem__(self, idx):
        a0=self.raw_data[idx:idx+block_size]
        b0=self.raw_data[idx+1:idx+block_size+1]
        a1=torch.tensor(a0)
        b1=torch.tensor(b0)
        return a1,b1


def text2token(char2index,text):
    n=len(text)
    a=torch.zeros((n,),dtype=torch.long)
    for i in range(n):
        ch=text[i]
        a[i]=char2index[ch]
    a=cut_pad(a)
    return a

def token2text(index2char,token):
    text=""
    for i in token:
        text+=index2char[i.item()]
    # text = text.replace(' ', '')
    return text

def on_batch_end_callback(trainer):
    if trainer.iter_num%100==0:
        print(f"iter_num={trainer.iter_num},loss={trainer.loss}")

if __name__ == '__main__':
    # files = ["my/text/sanguoyanyi.txt","my/text/hongloumeng.txt"]
    # files = ["my/text/tangshisanbaishou.txt"]

    # file='my/text/tangshisanbaishou.txt'
    file = "my/text/sanguoyanyi.txt"
    number=1
    char2index={' ':0}
    index2char={0:' '}
    with open(file, 'r') as f:
        for line in f:
            for char in line:
                if char in char2index:
                    continue
                else:
                    char2index[char] = number
                    index2char[number] = char
                    number += 1
    # sys.exit(0)
    # model_type = 'gpt2'
    model_type = 'gpt-nano'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")
    # sys.exit(0)
    dataset=Data(file, char2index,index2char)
    train_dataset = dataset

    model_config = GPT.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = len(char2index) + 1 #因为padding了0,所以这里多一个；
    model_config.block_size = block_size
    model = GPT(model_config)

    user_input=input("Train or not, Please enter 'y' or 'n': ")
    if user_input=='y':
        train=1
    else:
        train=0

    if train:
        train_config = Trainer.get_default_config()
        train_config.learning_rate = 5e-3 # many possible options, see the file
        train_config.max_iters = 10000
        train_config.batch_size = 32

        train_config.device=device
        train_config.weight_decay=1e-6
        train_config.betas=(0.9, 0.999)
        train_config.num_workers=1

        trainer = Trainer(train_config, model, train_dataset)
        trainer.set_callback('on_batch_end', on_batch_end_callback)
        trainer.run()

        # 假设模型的参数保存在变量 model 中
        torch.save(model.state_dict(), 'model.pth')
    else:
        # 加载模型的参数
        model.load_state_dict(torch.load('model.pth'))
        model.eval();
        while True:
            try:
                a=input(">>")
                # a='曹操'
                t=text2token(char2index,a)
                t=t.unsqueeze(0)
                out=model.generate(t, block_size)
                out.squeeze_(0)
                text=token2text(index2char,out)
                print(f"{text}\n")
            except Exception as e:
                print(e)



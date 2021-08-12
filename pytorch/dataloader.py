import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tqdm import tqdm

from pytorch.dataframe import get_dfs


def yield_tokens(df_data, tokenizer):
  for text in tqdm(df_data["titletext"]):
    yield tokenizer(text)


def text_pipeline(text, vocab, tokenizer):
  return vocab(tokenizer(text))


def df_to_dataset(df):
  df = df.copy()
  df = df[["label", "titletext"]]

  return list(df.to_records(index=False))


def collate_batch(batch, device, vocab, tokenizer):
  label_list, titletext_list, titletext_len_list = [], [], []

  def _text_pipeline(text): return text_pipeline(text, vocab, tokenizer)

  for label, titletext in batch:
    label_list.append(label)
    titletext = _text_pipeline(titletext)
    titletext = torch.tensor(titletext, dtype=torch.int64)
    titletext_list.append(titletext)
    titletext_len_list.append(len(titletext))

  label_list = torch.tensor(label_list, dtype=torch.float32)
  titletext_list = pad_sequence(
      titletext_list, batch_first=True)
  titletext_len_list = torch.tensor(titletext_len_list, dtype=torch.int64)

  return label_list.to(device), titletext_list.to(device), titletext_len_list.to(device)


def get_vocab(df_train, tokenizer):
  def _yield_tokens(df): return yield_tokens(df, tokenizer)

  vocab = build_vocab_from_iterator(
      _yield_tokens(df_train), specials=["<unk>"])
  vocab.set_default_index(vocab["<unk>"])

  return vocab


def get_dataloaders(base_df_dir, batch_size, device):
  df_train, df_valid, df_test = get_dfs(base_df_dir)
  tokenizer = get_tokenizer("basic_english")
  vocab = get_vocab(df_train, tokenizer)

  def _collate_fn(x): return collate_batch(x, device, vocab, tokenizer)

  train_iter = DataLoader(df_to_dataset(
      df_train), batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
  valid_iter = DataLoader(df_to_dataset(
      df_valid), batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
  test_iter = DataLoader(df_to_dataset(
      df_test), batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)

  return train_iter, valid_iter, test_iter


def debug():
  train_iter, _, _ = get_dataloaders("./data/", 2, torch.device("cpu"))
  counter = 0

  for labels, titletext, titletext_len in train_iter:
    print(labels, titletext, titletext_len)
    counter += 1

    if counter == 15:
      assert False
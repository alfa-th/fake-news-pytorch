import pandas as pd

from sklearn.model_selection import train_test_split

(TEST_RATIO, VALID_RATIO) = (0.80, 0.80)

def trim_df(base_dir):
  df_raw = pd.read_csv(base_dir + "news.csv")

  df_raw["label"] = (df_raw["label"] == "FAKE").astype("int")
  df_raw["titletext"] = df_raw["title"] + ". " + df_raw["text"]
  df_raw = df_raw.reindex(columns=["label", "title", "text", "titletext"])

  df_raw.drop(df_raw[df_raw.text.str.len() < 5].index, inplace=True)

  def trim_string(x):
    x = x.split(maxsplit=200)
    x = " ".join(x[:200])
    return x

  df_raw["text"] = df_raw["text"].apply(trim_string)
  df_raw["titletext"] = df_raw["titletext"].apply(trim_string)

  df_real = df_raw[df_raw["label"] == 0]
  df_fake = df_raw[df_raw["label"] == 1]

  return df_real, df_fake


def split_df(df_real, df_fake):
  df_real_trainvalid, df_real_test = train_test_split(
      df_real, train_size=TEST_RATIO, random_state=1)
  df_fake_trainvalid, df_fake_test = train_test_split(
      df_fake, train_size=TEST_RATIO, random_state=1)

  df_real_train, df_real_valid = train_test_split(
      df_real_trainvalid, train_size=VALID_RATIO, random_state=1)
  df_fake_train, df_fake_valid = train_test_split(
      df_fake_trainvalid, train_size=VALID_RATIO, random_state=1)

  df_train = pd.concat([df_real_train, df_fake_train],
                       ignore_index=True, sort=False)
  df_valid = pd.concat([df_real_valid, df_fake_valid],
                       ignore_index=True, sort=False)
  df_test = pd.concat([df_real_test, df_fake_test],
                      ignore_index=True, sort=False)

  return df_train, df_valid, df_test

def write_dfs(df_train, df_valid, df_test, base_dir):
  df_train.to_csv(base_dir + "news_train.csv", index=False)
  df_valid.to_csv(base_dir + "news_valid.csv", index=False)
  df_test.to_csv(base_dir + "news_test.csv", index=False)

def get_dfs(base_dir):
  try:
    df_train = pd.read_csv(base_dir + "news_train.csv")
    df_valid = pd.read_csv(base_dir + "news_valid.csv")
    df_test = pd.read_csv(base_dir + "news_test.csv")
  except Exception:
    df_real, df_fake = trim_df(base_dir)
    dfs = split_df(df_real, df_fake)
    write_dfs(*dfs, base_dir)

    df_train, df_valid, df_test = dfs

  return df_train, df_valid, df_test

def debug():
  df_train, _, _ = get_dfs("./data/")

  df_train

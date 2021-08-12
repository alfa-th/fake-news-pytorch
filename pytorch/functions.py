import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train(
    model,
    optimizer,
    device,
    train_loader,
    valid_loader,
    criterion=nn.BCELoss(),
    num_epochs=5,
    eval_every=5,
):
  run_loss = 0.0
  valid_run_loss = 0.0
  global_step = 0
  train_loss_list = []
  valid_loss_list = []
  global_step_list = []
  ave_train_loss = None
  ave_valid_loss = None

  model.train()
  for epoch in range(num_epochs):
    with tqdm(train_loader, unit="batch") as tbatch:
      for labels, titletext, titletext_len in tbatch:
        tbatch.set_description(f"Epoch {epoch}")

        output = model(titletext, titletext_len)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        global_step += 1

        tbatch.set_postfix(step=f"{ global_step }/{(num_epochs * len(train_loader)) }",
                           train_loss=ave_train_loss,
                           valid_loss=ave_valid_loss)

        if global_step % eval_every == 0:
          model.eval()
          with torch.no_grad():
            for labels, titletext, titletext_len in valid_loader:
              labels = labels.to(device)
              titletext = titletext.to(device)
              titletext_len = titletext_len.to(device)

              output = model(titletext, titletext_len)
              loss = criterion(output, labels)

              valid_run_loss += loss.item()

          ave_train_loss = run_loss / eval_every
          ave_valid_loss = run_loss / len(valid_loader)
          train_loss_list.append(ave_train_loss)
          valid_loss_list.append(ave_valid_loss)
          global_step_list.append(global_step)

          run_loss, valid_run_loss = 0.0, 0.0

          model.train()
          tbatch.set_postfix(step=f"{ global_step }/{(num_epochs * len(train_loader)) }",
                             train_loss=loss.item(),
                             valid_loss=ave_valid_loss)

  plot_train(train_loss_list, valid_loss_list, global_step_list)
  print("Finished training")

def plot_train(train_loss_list, valid_loss_list, global_steps_list):
  plt.plot(global_steps_list, train_loss_list, label="Train")
  plt.plot(global_steps_list, valid_loss_list, label="Train")

  plt.xlabel("Global steps")
  plt.ylabel("Loss")

  plt.legend()
  plt.show()


def eval(model, test_loader, device, threshold):
  y_pred = []
  y_true = []

  model.eval()

  with torch.no_grad():
    with tqdm(test_loader, unit="batch") as tbatch:
      for labels, titletext, titletext_len in tbatch:
        output = model(titletext, titletext_len)
        output = (output > threshold).int()

        y_pred.extend(output.to_list())
        y_true.extend(labels.to_list())
  
  plot_eval(y_true, y_pred)
    

def plot_eval(y_true, y_pred):
  print("Classification Report")
  print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

  cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

  ax = plt.subplot()
  sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="d")
  ax.set_title("Confusion Matrix")
  ax.set_xlabel("Predicted Labels")
  ax.set_ylabel("True Labels")

  ax.xaxis.set_ticklabels(["FAKE", "REAL"])
  ax.yaxis.set_ticklabels(["FAKE", "REAL"])
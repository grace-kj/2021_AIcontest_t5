from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset import SKT_BoolQ, T5_Classification_Collator
from torch.utils.data import DataLoader
from transformers import AdamW, get_constant_schedule_with_warmup
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

random_seed = 42

random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Load T5 model...")
tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-large")
model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-large")
model.cuda()

batch_size = 1

collator = T5_Classification_Collator(use_tokenizer=tokenizer, max_sequence_len=512)

print("Load Dataset...")
train_dataset = SKT_BoolQ('SKT_BoolQ_Train.tsv')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

dev_dataset = SKT_BoolQ('SKT_BoolQ_Dev.tsv')
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

epochs = 10
total_steps = len(train_dataloader) * epochs

optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )
#scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                            num_warmup_steps = len(train_dataloader), # Default value in run_glue.py
#                                            num_training_steps = total_steps)
scheduler = get_constant_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = len(train_dataloader))

print("Training...")

for epoch_i in range(epochs):
  print("-------------------------------------------------------------")
  # train
  model.train()

  true_labels = []
  prediction_labels = []
  total_loss = 0

  for batch in tqdm(train_dataloader):
    true_labels += batch['labels'].numpy().flatten().tolist()
    batch = {k:v.type(torch.long).cuda() for k,v in batch.items()}

    model.zero_grad()
    outputs = model(**batch)

    loss, logits = outputs[:2]
    total_loss += loss.item()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    logits = logits.detach().cpu().numpy()

    prediction_labels += logits.argmax(axis=-1).flatten().tolist()

  train_avg_epoch_loss = total_loss / len(train_dataloader)
  train_acc = accuracy_score(true_labels, prediction_labels)

  prediction_labels = []
  true_labels = []
  total_loss = 0

  model.eval()

  # Evaluate data for one epoch
  for batch in tqdm(dev_dataloader):
    true_labels += (batch['labels'].numpy())[:, 0].flatten().tolist()
    batch = {k:v.type(torch.long).cuda() for k,v in batch.items()}

    with torch.no_grad():        
        outputs = model(**batch)

        loss, logits = outputs[:2]
        
        logits = logits.detach().cpu().numpy()
        total_loss += loss.item()

        predict_content = (logits.argmax(axis=-1))[:, 0].flatten().tolist()
        prediction_labels += predict_content

  # Calculate the average loss over the training data.
  dev_avg_epoch_loss = total_loss / len(dev_dataloader)
  #print(len(true_labels))
  #print(len(prediction_labels))
  #print(len(dev_dataloader))
  dev_acc = accuracy_score(true_labels, prediction_labels)

  print("Epoch: %d  train_loss: %.5f  train_acc: %.5f  dev_loss: %.5f  dev_acc: %.5f"%(epoch_i, train_avg_epoch_loss, train_acc, dev_avg_epoch_loss, dev_acc))

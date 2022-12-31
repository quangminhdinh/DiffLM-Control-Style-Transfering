from transformers import Trainer, TrainingArguments
from .config import model, train_set, val_set
from .utils import compute_metrics


training_args = TrainingArguments(
    output_dir='./checkpoints/sdpe-250',          # output directory
    num_train_epochs=20,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",
    eval_accumulation_steps=10,
    label_names=['labels'],
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_set,         # training dataset
    eval_dataset=val_set,             # evaluation dataset
    compute_metrics=compute_metrics,
)

from transformers import ViTModel, ViTConfig, ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
import torch
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D
import numpy as np
from transformers import default_data_collator
from transformers import ViTModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class ViTForImageClassification2(nn.Module):

    def __init__(self, num_labels=10):

        super(ViTForImageClassification2, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):

        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs['pooler_output']) # Have to take the pooler output. Other option is last hidden state 
        loss = None

        if labels is not None:

          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

train_ds, test_ds = load_dataset('cifar10', split=['train[:1000]', 'test[:200]']) 
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']
metric = load_metric("accuracy")
data_collator = default_data_collator

model = ViTForImageClassification2()
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def preprocess_images(examples):

    images = examples['img']
    images = [np.array(image, dtype=np.uint8) for image in images]
    images = [np.moveaxis(image, source=-1, destination=0) for image in images]
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']

    return examples

features = Features({
    'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
    'img': Array3D(dtype="int64", shape=(3,32,32)),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})

preprocessed_train_ds = train_ds.map(preprocess_images, batched=True, features=features)
preprocessed_val_ds = val_ds.map(preprocess_images, batched=True, features=features)
preprocessed_test_ds = test_ds.map(preprocess_images, batched=True, features=features)

# print(f" type is : {type(preprocessed_train_ds)} , print : {preprocessed_train_ds})")

model.train()

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

args = TrainingArguments(
    "test-cifar-10",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model='accuracy',
)

trainer = Trainer(
    model,
    args,
    train_dataset = preprocessed_train_ds,
    eval_dataset = preprocessed_val_ds,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)

trainer.train()
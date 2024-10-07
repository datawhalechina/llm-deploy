from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import json
import datasets
from torch.utils.data import DataLoader

def load_finetune_data(path):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data
    else:
        return None
    
def tokenize(example):
    if model_type == "decoder":
        # Tokenize and apply left side padding manually

        # Tokenize in vanilla Python list form
        it = tokenizer(
            example["input"],
            max_length=512,
            truncation=True
        )
        iids = it["input_ids"]
        if "label" in example:
            lids = tokenizer(
                example["label"],
                max_length=512,
                truncation=True
            )["input_ids"]
        else:
            lids = [list() for _ in range(len(iids))]

        lengths = []
        input_ids = []
        attention_mask = []
        label_ids = []
        for iid, lid in zip(iids, lids):
            lengths.append(len(iid) + len(lid))
            input_ids.append(iid + lid)
            attention_mask.append([1] * (len(iid) + len(lid)))
            label_ids.append([-100] * len(iid) + lid)

        # Pad full sequences
        lengths = torch.tensor(lengths)
        pad_lengths = (lengths.max() - lengths).tolist()
        for i, l in enumerate(pad_lengths):
            # Apply left side padding
            # Why? https://github.com/huggingface/transformers/issues/3021#issuecomment-1231526631
            input_ids[i] = [tokenizer.pad_token_id] * l + input_ids[i]
            attention_mask[i] = [0] * l + attention_mask[i]
            label_ids[i] = [-100] * l + label_ids[i]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
    else:
        raise NotImplementedError(model_type)
    
def train_step(batch):
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }
    res = model(**kwargs)["loss"]
    return res

def dataloader(path):
    data = load_finetune_data(path)
    dataset = datasets.Dataset.from_dict(data)
    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(dataset, batch_size=8, shuffle=True)

def val():
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            loss = train_step(batch)
            print("Val Step: {}, Loss: {}".format(step, loss.item()))
def train():
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(20):
        for step, batch in enumerate(train_dataloader):
            loss = train_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 4 == 0:
                print("Step: {}, Loss: {}".format(step, loss.item()))
                torch.save(model.state_dict(), "model_step_{}.pt".format(i))
                val()

def test(ckpt_path=""):
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }
            res = model(**kwargs)
            loss = res["loss"]
            print("Test Step: {}, Loss: {}".format(step, loss.item()))

            # todo: print output
                      


if __name__ == "__main__":
    model_name_or_path = "F:\llm-deploy\GPT2_deploy\GPT-2"
    model_type = "decoder"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    train_data_path = r"F:\llm-deploy\GPT2_deploy\data\F_zs_cot_date_understanding_good_train.jsonl"
    val_data_path = r"F:\llm-deploy\GPT2_deploy\data\F_zs_cot_date_understanding_good_val.jsonl"
    test_data_path = r"F:\llm-deploy\GPT2_deploy\data\F_zs_cot_date_understanding_good_test.jsonl"

    train_dataloader = dataloader(train_data_path)
    val_dataloader = dataloader(val_data_path)
    test_dataloader = dataloader(test_data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train()
    test()


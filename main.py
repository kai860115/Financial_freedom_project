import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

class RandomSentencesDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=self.max_length, padding="max_length")
        return torch.tensor(inputs, dtype=torch.long)

def train_model(num_epochs, learning_rate, model_name, training_sentences, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    dataset = RandomSentencesDataset(training_sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            inputs = batch.to(model.device)
            labels = inputs.clone()

            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

        sentences = generate_sentences(model, tokenizer, training_sentences[-1][31:], num_sentences=5, max_length=200, do_sample=True, temperature=1.0)
        for s in sentences:
            print(s)

    return model, tokenizer


def generate_sentences(model, tokenizer, input_text, num_sentences=5, max_length=200, do_sample=True, num_beams=5, top_k=0, top_p=0.9, temperature=1.0):
    model.eval()
    generated_sentences = []
    for _ in range(num_sentences):
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_sentences.append(generated_sentence)

    return generated_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or generate sentences using GPT-2")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"], help="Specify whether to train the model or generate sentences")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--train_data_path", type=str, default="lotto649.json", help="training data")
    parser.add_argument("--prev_num", type=int, default=5, help="prev lottery num")

    # Inference parameters
    parser.add_argument("-n", "--num_sentences", type=int, default=1, help="Number of sentences to generate")
    parser.add_argument("--num_beams", type=int, default=5, help="For beam search")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of the generated sentences")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling during sentence generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature during sentence generation")
    parser.add_argument("--date", type=str, default="112/07/18,34,19,04,40,05,16,10|112/07/21,03,15,35,41,49,39,38|112/07/25,17,33,40,12,28,45,39|112/07/28,03,22,25,34,44,01,32", help="date")
    args = parser.parse_args()

    if args.mode == "train":
        # Training
        df = pd.read_json(args.train_data_path)
        df['獎號'] = df['獎號'].apply(lambda x: ','.join(str(e) for e in x))
        df['特別號'] = df['特別號'].apply(lambda x: str(x) if x >= 10 else '0' + str(x))
        sentences = df[["開獎日期" ,"獎號", "特別號"]].apply(",".join, axis=1).tolist()
        sentences.sort()

        training_sentences = []

        for i in range(len(sentences) - args.prev_num + 1):
            training_sentences.append("|".join(sentences[i : i + args.prev_num]))

        for i in range(5):
            print(training_sentences[i])

        model, tokenizer = train_model(args.num_epochs, args.learning_rate, args.model_name, training_sentences, args.batch_size)

        # Save the model
        os.makedirs("ckpt", exist_ok=True)
        model_save_path = os.path.join("ckpt", args.model_name)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

    elif args.mode == "generate":
        # Load the model
        model_save_path = os.path.join("ckpt", args.model_name)
        model = AutoModelForCausalLM.from_pretrained(model_save_path)
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        model.to(device)

        # Generate sentences using the loaded model with specified inference parameters
        generated_sentences = generate_sentences(
            model,
            tokenizer,
            args.date,
            num_sentences=args.num_sentences,
            max_length=args.max_length,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            top_k=args.k,
            top_p=args.p,
            temperature=args.temperature
        )

        # Output the results
        print("=== Generated Sentences ===")
        for sentence in generated_sentences:
            print(sentence)

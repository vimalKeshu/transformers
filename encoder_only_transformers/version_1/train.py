import os
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertTokenizerFast
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from model import EncoderOnlyTransformer
from news_dataset import NewsDataset
from pathlib import Path

current_directory = os.path.dirname(os.path.abspath(__file__))

def train(model, train_loader, criterion, optimizer, device, epoch, cfg):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad() # reset the gradients
        outputs = model(input_ids, masks) # predict the output, forward pass
        loss = criterion(outputs, labels) # compute the loss
        loss.backward() # backpropagate the loss, calculate the gradient
        optimizer.step() # update the weights
        total_loss += loss.item() # calculate the total loss
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")
    return avg_loss

def evaluate(model, test_loader, device):
    torch.cuda.empty_cache()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, masks)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total * 100
    wandb.log({"test_accuracy": accuracy})
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project_name)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True)) # Convert Hydra DictConfig to a standard dictionary for wandb

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_ms or torch.backends.mps.is_available else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device=device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    dataset = load_dataset(cfg.dataset.name)
    tokenizer:BertTokenizerFast = BertTokenizerFast.from_pretrained(cfg.model.tokenizer, cache_dir="./cache")

    train_ds = NewsDataset(ds=dataset["train"], tokenizer=tokenizer, max_seq_len=cfg.model.max_seq_len)
    test_ds = NewsDataset(ds=dataset["test"], tokenizer=tokenizer, max_seq_len=cfg.model.max_seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.test.batch_size)
    
    model = EncoderOnlyTransformer(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        num_layers=cfg.model.num_layers,
        num_classes=cfg.model.num_classes,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr)
    
    checkpoint_path = os.path.join(current_directory, cfg.training.checkpoint_dir_name)
    Path(f"{checkpoint_path}").mkdir(parents=True, exist_ok=True)
    latest_checkpoint = None
    initial_epoch = 0
    global_step = 0    

    if os.path.exists(checkpoint_path):
        latest_checkpoint = max([os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path)], default=None)
        if latest_checkpoint:
            print(f"Loading checkpoint from {latest_checkpoint}")
            state = torch.load(latest_checkpoint)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1 
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step'] 
    else:
        print('No model to preload, starting from scratch')

    for epoch in range(initial_epoch, cfg.training.epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, cfg)
        evaluate(model, test_loader, device)
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, checkpoint_file)            
            print(f"Checkpoint saved at {checkpoint_file}")
        global_step += 1
    wandb.finish()

if __name__ == "__main__":
    main()

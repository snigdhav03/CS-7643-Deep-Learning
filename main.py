import typer
import torch
from torch.utils.data import DataLoader

from your_dataset import CustomDataset
from your_model import DeepLearningModel

app = typer.Typer()  

@app.command()
def train(
    data_path: str = typer.Argument(..., help="Path to the dataset directory"),
    model_arch: str = typer.Option("resnet18", help="Model architecture to use"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    epochs: int = typer.Option(5, help="Number of training epochs"),
    learning_rate: float = typer.Option(0.001, help="Learning rate for the optimizer")
):
    dataset = CustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepLearningModel(model_arch)
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for _ in range(epochs):
        for _, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
 

if __name__ == "__main__":
    app()

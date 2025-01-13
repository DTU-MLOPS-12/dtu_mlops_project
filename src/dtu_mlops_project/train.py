import matplotlib.pyplot as plt
import torch
import typer
import wandb
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

from model import MyAwesomeModel
from data import corrupt_mnist
from utils import get_device

DEVICE = get_device()

app = typer.Typer()


@app.command()
def train(
    lr: float = typer.Option(1e-3, help="Learning rate for the optimizer"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    epochs: int = typer.Option(10, help="Number of training epochs"),
) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads)})

    print("Training complete")
    model_path = "models/model.pth"
    torch.save(model.state_dict(), model_path)

    # Log the model as an artifact
    artifact = wandb.Artifact(
        name="corrupt_mnist_model", type="model", description="A model trained to classify corrupt MNIST images"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # Link the artifact to the model registry
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    app()

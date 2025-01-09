import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Annotated

from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

@app.command()
def visualize(
    model_checkpoint: Annotated[Path, typer.Option("--model-checkpoint", help="Path to the model checkpoint file")],
    figure_name: Annotated[str, typer.Option("embeddings.png", help="Name of the output figure file")]
) -> None:
    """Visualize model predictions."""

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    model.eval()

    model.fc1 = torch.nn.Identity()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            images, target = images.to(DEVICE), target.to(DEVICE)

            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)

        embeddings = torch.cat(embeddings).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    app()

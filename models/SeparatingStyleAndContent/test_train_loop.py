
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt


def train_test_routine(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lossfn,
        trainloader: DataLoader,
        testloader: DataLoader,
        device,
        epochs: int = 100,
        state_directory: str | Path | None = "model_files",
        save_each: int = 4,
        start_from_saved_state: bool = False,
        show_graphs: bool = True
):
    if start_from_saved_state:
        model.load_state_dict(torch.load(state_directory + "/model.pth"))
        optimizer.load_state_dict(torch.load(state_directory + "/optimizer.pth"))

    train_per_batch_losses = []
    test_per_batch_losses = []

    def loop():
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = torch.tensor(0., dtype=torch.double)
            train_batch = 0.
            for (content_b, style_b), target_b in tqdm(iterable=trainloader,
                                                       desc=f"Epoch {epoch}: train") if show_graphs else trainloader:
                content_b, style_b, target_b = content_b.to(device), style_b.to(device), target_b.to(device)
                optimizer.zero_grad()
                output_b = model(content_b, style_b)
                loss = lossfn(target_b, output_b)
                train_loss += loss.to("cpu")
                train_batch += 1
                loss.backward()
                optimizer.step()
            train_per_batch_losses.append((train_loss / train_batch).item())

            model.eval()
            test_loss = torch.tensor(0., dtype=torch.double)
            test_batch = 0.
            with torch.no_grad():
                for (content_b, style_b), target_b in tqdm(iterable=testloader,
                                                           desc=f"Epoch {epoch}: test") if show_graphs else testloader:
                    content_b, style_b, target_b = content_b.to(device), style_b.to(device), target_b.to(device)
                    output_b = model(content_b, style_b)
                    loss = lossfn(target_b, output_b)
                    test_loss += loss.to("cpu")
                    test_batch += 1
            test_per_batch_losses.append((test_loss / test_batch).item())

            if save_each != -1 and epoch % save_each == 0:
                if type(state_directory) == str:
                    state_path = Path(state_directory)
                else:
                    state_path = state_directory
                if not Path.is_dir(state_path):
                    state_path.mkdir()
                torch.save(obj=model, f=state_path / "model.pth")
                torch.save(obj=optimizer, f=state_path / "optimizer.pth")

            if show_graphs:
                clear_output()
                _, (sp1, sp2, sp3) = plt.subplots(1, 3)
                with torch.no_grad():
                    for (content_b, style_b), target_b in tqdm(iterable=testloader,
                                                               desc=f"Epoch {epoch}: test") if show_graphs else testloader:
                        content_b, style_b, target_b = content_b.to(device), style_b.to(device), target_b.to(device)
                        output_b = model(content_b, style_b)
                        sp2.imshow(output_b[0].permute(1, 2, 0).to("cpu"), cmap='gray')
                        sp3.imshow(target_b[0].permute(1, 2, 0).to("cpu"), cmap='gray')
                        break
                sp1.plot(np.arange(1, epoch + 1), test_per_batch_losses, color='r', label='test')
                sp1.plot(np.arange(1, epoch + 1), train_per_batch_losses, color='g', label='train')
                sp1.legend()
                plt.show()

    loop()
    return train_per_batch_losses, test_per_batch_losses

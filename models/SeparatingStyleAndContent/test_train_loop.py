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
        save_each: int = 1,
        start_from_saved_state: bool = False,
        show_graphs: bool = True,
        n_checkpoints: int = None
):
    if start_from_saved_state:
        model = torch.load(state_directory + "/model.pth")
        optimizer = torch.load(state_directory + "/optimizer.pth")

    def save_progress(epoch: int):
        if save_each != -1 and epoch % save_each == 0:
            if type(state_directory) == str:
                state_path = Path(state_directory)
            else:
                state_path = state_directory
            if not Path.is_dir(state_path):
                state_path.mkdir()
            torch.save(obj=model, f=state_path / "model.pth")
            torch.save(obj=optimizer, f=state_path / "optimizer.pth")

    save_progress(0)

    train_per_batch_losses = []
    test_per_batch_losses = []

    if n_checkpoints is None:
        n_checkpoints = 1

    n_train_batches = len(trainloader)
    n_test_batches = len(testloader)

    train_check_period = n_train_batches // n_checkpoints
    test_check_period = n_test_batches // n_checkpoints

    def train_epoch(epoch: int):
        train_loss = torch.tensor(0., dtype=torch.double)
        train_batch = 0
        for (content_b, style_b), target_b in tqdm(iterable=trainloader,
                                                   desc=f"Epoch {epoch}: train") if show_graphs else trainloader:
            model.train()

            torch.set_grad_enabled(True)
            content_b, style_b, target_b = content_b.to(device), style_b.to(device), target_b.to(device)
            output_b = model(content_b, style_b)
            loss = lossfn(target_b, output_b)
            loss.backward()
            train_loss += loss.to("cpu")
            train_batch += 1
            optimizer.step()
            optimizer.zero_grad()
            if train_batch % train_check_period == 0:
                train_per_batch_losses.append((train_loss / train_check_period).item())
                train_loss = torch.tensor(0., dtype=torch.double)
                yield
            elif train_batch == n_train_batches:
                train_per_batch_losses.append((train_loss / (n_train_batches % train_check_period)).item())
                yield

    def test_epoch(epoch: int):
        test_loss = torch.tensor(0., dtype=torch.double)
        test_batch = 0
        for (content_b, style_b), target_b in tqdm(iterable=testloader,
                                                   desc=f"Epoch {epoch}: test") if show_graphs else testloader:
            model.eval()
            torch.set_grad_enabled(False)

            def show_sample_and_graph():
                if show_graphs:
                    clear_output()
                    _, (sp1, sp2, sp3) = plt.subplots(1, 3)
                    sp2.imshow(output_b[0].permute(1, 2, 0).to("cpu"), cmap='gray')
                    sp3.imshow(target_b[0].permute(1, 2, 0).to("cpu"), cmap='gray')
                    sp1.plot(np.arange(len(test_per_batch_losses)), test_per_batch_losses, color='r', label='test')
                    sp1.plot(np.arange(len(test_per_batch_losses)), train_per_batch_losses, color='g',
                             label='train')
                    sp1.legend()
                    plt.show()

            content_b, style_b, target_b = content_b.to(device), style_b.to(device), target_b.to(device)
            output_b = model(content_b, style_b)
            loss = lossfn(target_b, output_b)
            test_loss += loss.to("cpu")
            test_batch += 1
            if test_batch % test_check_period == 0:
                test_per_batch_losses.append((test_loss / test_check_period).item())
                test_loss = torch.tensor(0., dtype=torch.double)
                show_sample_and_graph()
                yield
            elif test_batch == n_test_batches:
                test_per_batch_losses.append((test_loss / (n_test_batches % test_check_period)))
                show_sample_and_graph()
                yield

    def loop():
        for epoch in range(1, epochs + 1):
            train_gen = train_epoch(epoch)
            test_gen = test_epoch(epoch)
            try:
                while True:
                    train_gen.__next__()
                    test_gen.__next__()
            except StopIteration:
                pass

            save_progress(epoch)

    loop()
    return train_per_batch_losses, test_per_batch_losses

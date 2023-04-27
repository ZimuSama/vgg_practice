import torch
# define training and testing function
def train(dataloader, model, loss_fn, metric_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    loss_sum = 0.0
    metric_sum = 0.0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        metric = metric_fn(pred,y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Caculate loss and accuracy
        loss_sum += loss.item()
        metric_sum += metric.item()
    loss_sum /= num_batches
    metric_sum /= num_batches
    print(f"loss: {loss_sum:>7f}, acc: {metric_sum:>7f} [train]")
    return metric_sum, loss_sum
def test(dataloader, model, loss_fn, metric_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss_sum = 0.0
    metric_sum = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_sum += loss_fn(pred, y).item()
            metric_sum += metric_fn(pred,y).item()
    loss_sum /= num_batches
    metric_sum /= num_batches
    print(f"loss: {loss_sum:>7f}, acc: {metric_sum:>7f} [test]")
    return metric_sum, loss_sum

def save(model, path, partly = False):
    state_dict = model.state_dict()
    if partly:
        for param_tensor in list(state_dict):
            if not 'conv' in param_tensor:
                state_dict.pop(param_tensor)
    torch.save(state_dict, path)
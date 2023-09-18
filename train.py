import torch
from utils import AverageMeter
from tqdm import tqdm

root = "nyu_depth_v2_labeled.mat"


train_augmentations = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop((240, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(50),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1, 1)),
    transforms.ElasticTransform(alpha=25.0, sigma=5.0),
])

test_augmentations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((240, 320)),
])

train_nyu_dataset = NyuDataset(root, augmentations=train_augmentations, normalize=True, depth_norm=10)
test_nyu_dataset = NyuDataset(root, augmentations=test_augmentations, normalize=True, depth_norm=10)



from torch.utils.data import random_split

seed = 42

# get train split
num_train = round(0.7*len(train_nyu_dataset))
num_remain = round(0.3*len(train_nyu_dataset))
(train_dataset, _) = random_split(train_nyu_dataset,
                                              [num_train, num_remain],
                                              generator=torch.Generator().manual_seed(seed))

""" 
Sketchy hack to get valid/test datasets
"""

(_, remain_dataset) = random_split(test_nyu_dataset,
                                              [num_train, num_remain],
                                              generator=torch.Generator().manual_seed(seed))

# get valid and test split
num_valid = round(0.8*len(remain_dataset))
num_test = round(0.2*len(remain_dataset))
(valid_dataset, test_dataset) = random_split(remain_dataset,
                                             [num_valid, num_test],
                                              generator=torch.Generator().manual_seed(seed))


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) 
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)



def train(model, opts, crits, dataloader, loss_coeffs=(1.0,), grad_norm=0.0):
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for sample in pbar:
        loss = 0.0
        input = #TODO: Get the Input
        targets = #TODO: Get the Targets
        
        #FORWARD
        outputs = #TODO: Run a Forward pass

        for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
            #TODO: Increment the Loss

        # BACKWARD
        #TODO: Zero Out the Gradients
        #TODO: Call Loss.Backward

        if grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        #TODO: Run one step

        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )

def validate(model, metrics, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for metric in metrics:
        metric.reset()

    pbar = tqdm(dataloader)

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    with torch.no_grad():
        for sample in pbar:
            # Get the Data
            input = sample["image"].float().to(device)
            targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]

            #input, targets = get_input_and_targets(sample=sample, dataloader=dataloader, device=device)
            targets = [target.squeeze(dim=1).cpu().numpy() for target in targets]

            # Forward
            outputs = model(input)
            #outputs = make_list(outputs)

            # Backward
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
                    .squeeze(dim=1)
                    .cpu()
                    .numpy(),
                    target,
                )
            pbar.set_description(get_val(metrics)[1])
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals
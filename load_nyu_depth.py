

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

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

import numpy as np
import random
import h5py
from torchvision import models
from sklearn.metrics import confusion_matrix
import argparse
import os

class TriDataset(data.Dataset):
    """Custom Dataset for loading and processing the data."""

    def __init__(self, features, labels, source, transform=None):
        self.features = features
        self.labels = labels
        self.source = source
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        anchor = self.features[index]
        anchor_label = self.labels[index]

        # positive
        pos_index = random.randint(0, len(self.labels) - 1)
        while self.labels[pos_index] != anchor_label:
            pos_index = random.randint(0, len(self.labels) - 1)
        positive = self.features[pos_index]
        positive_label = self.labels[pos_index]

        # negative
        neg_index = random.randint(0, len(self.labels) - 1)
        while self.labels[neg_index] == anchor_label:
            neg_index = random.randint(0, len(self.labels) - 1)
        negative = self.features[neg_index]
        negative_label = self.labels[neg_index]

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return torch.from_numpy(anchor).float(), \
            torch.from_numpy(positive).float(), \
            torch.from_numpy(negative).float(), \
            torch.tensor(anchor_label, dtype=torch.long), \
            torch.tensor(positive_label, dtype=torch.long), \
            torch.tensor(negative_label, dtype=torch.long)

    def get_source(self):
        return self.source

class ActTriNet(nn.Module):
    def __init__(self, act):
        super(ActTriNet, self).__init__()
        self.act = act

    def forward(self, x, act_dim):

        x_activation= x[:, :, :, :act_dim]
        x = self.act(x_activation)
        embedding = F.normalize(x, p=2, dim=1)

        return embedding

def train_and_evaluate_model(model, train_loader, test_loader, support_loader, save_path, act_dim, squeeze_dim=1, epochs=60):

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    highest_acc = 0
    test_accuracy_array = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (anchor, positive, negative, _, _, _) in enumerate(train_loader):

            anchor = anchor.unsqueeze(squeeze_dim).cuda()
            # print('!!!!!anchor:', anchor.shape)
            positive = positive.unsqueeze(squeeze_dim).cuda()
            negative = negative.unsqueeze(squeeze_dim).cuda()

            optimizer.zero_grad()
            anchor_embedding = model(anchor, act_dim)
            positive_embedding = model(positive, act_dim)
            negative_embedding = model(negative, act_dim)
            loss = criterion(anchor_embedding,
                             positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, train_loss))
        test_accuracy = test_model(model, test_loader, support_loader, act_dim, squeeze_dim=1)
        test_accuracy_array.append(test_accuracy)
        if test_accuracy > highest_acc:
            highest_acc = test_accuracy
            torch.save(model, save_path)
    print('===test_accuracy_array===:', test_accuracy_array)

def test_model(model, test_loader, support_loader, act_dim, squeeze_dim=1):

    model.eval()
    correct = 0
    y_pred = []
    y_true = []

    support_set_labels = []
    support_set_output = []
    with torch.no_grad():
        for i, (support_data, _, _, support_label, _, _) in enumerate(support_loader):
            support_data = support_data.unsqueeze(squeeze_dim).cuda()
            if i == 0:
                support_set_output = model(support_data, act_dim)
                support_set_labels = support_label
            else:
                support_set_output = torch.cat(
                    (support_set_output, model(support_data, act_dim)), dim=0)
                support_set_labels = torch.cat(
                    (support_set_labels, support_label), dim=0)

    # compare the distance between the embedding of the test image and the embedding of the support set
    with torch.no_grad():
        for i, (anchor, _, _, anchor_label, _, _) in enumerate(test_loader):
            anchor = anchor.unsqueeze(squeeze_dim).cuda()
            anchor_embedding = model(anchor, act_dim)
            anchor_embedding = anchor_embedding.squeeze()
            # if isinstance(support_set_output, list):
            #     support_set_output = torch.tensor(support_set_output)
            dist = F.pairwise_distance(
                anchor_embedding, support_set_output, p=2)
            pred = support_set_labels[torch.argmin(dist, -1)]
            y_pred.append(int(pred))
            y_true.append(int(anchor_label))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = conf_matrix.ravel()
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    print(f"True Positives: {TP / (TP + FN)}, False Positives: {FP / (FP + TN)}, True Negatives: {TN / (TN + FP)}, False Negatives: {FN / (TP + FN)}")
    # print(f"Confusion Matrix:{conf_matrix}")
    # accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    print(f"Accuracy: {accuracy}")
    return accuracy

def load_data(dataset_name, model_name='openai-community/gpt2-xl'):
    largest_data_num = {'openai-community/gpt2-xl': 10000, 'meta-llama/Llama-2-7b-hf': 10000, 'Mistral-7B': 8000, 'Llama3-8b': 8000, 'vicuna-7b-v1.5': 10000,
                        'stablelm-tuned-alpha-7b': 10000, 'Llama-2-13b': 8000, 'vicuna-13b-v1.5': 8000}
    for i, current_dataset_name in enumerate(dataset_name):
        print('./features/{}/all_data_{}.h5'.format(model_name, current_dataset_name))
        with h5py.File('./features/{}/all_data_{}.h5'.format(model_name, current_dataset_name), 'r') as f:
            activation_values = torch.tensor(
                f['all_activation_values'][:].astype(np.float32))
            label = torch.tensor(f['all_label'][:])
            source = torch.ones_like(label) * i
            print('shape of activation_values:', activation_values.shape)
            if current_dataset_name == dataset_name[0]:
                if activation_values.shape[0] > largest_data_num[model_name]:
                    all_activation_values = torch.cat((activation_values[:int(
                        largest_data_num[model_name]/2)], activation_values[-int(largest_data_num[model_name]/2):]), dim=0)
                    print(all_activation_values.shape[0])
                    all_label = torch.cat((label[:int(
                        largest_data_num[model_name]/2)], label[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_source = torch.cat((source[:int(
                        largest_data_num[model_name]/2)], source[-int(largest_data_num[model_name]/2):]), dim=0)
                else:
                    all_activation_values = activation_values
                    print(all_activation_values.shape[0])
                    all_label = label
                    all_source = source
            else:
                if activation_values.shape[0] > largest_data_num[model_name]:
                    all_activation_values = torch.cat((all_activation_values, activation_values[:int(
                        largest_data_num[model_name]/2)], activation_values[-int(largest_data_num[model_name]/2):]), dim=0)
                    print(all_activation_values.shape[0])
                    all_label = torch.cat((all_label, label[:int(
                        largest_data_num[model_name]/2)], label[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_source = torch.cat((all_source, source[:int(
                        largest_data_num[model_name]/2)], source[-int(largest_data_num[model_name]/2):]), dim=0)
                else:
                    all_activation_values = torch.cat(
                        (all_activation_values, activation_values), dim=0)
                    print(all_activation_values.shape[0])
                    all_label = torch.cat((all_label, label), dim=0)
                    all_source = torch.cat((all_source, source), dim=0)

    return all_activation_values, all_label, all_source

def split_set(data, label, source, support_size, ratio=[0.8, 0.2]):

    all_dataset = TriDataset(data, label, source)
    train_size = int(data.shape[0] * ratio[0])
    # test_size = data.shape[0] - train_size - support_size
    test_size = int(data.shape[0] * ratio[1])
    remain_size = data.shape[0] - train_size - test_size - support_size
    print('train size: {}, test size: {}, support size: {}'.format(
        train_size, test_size, support_size))
    train_data, test_data, support_data, remain_data = torch.utils.data.random_split(
        all_dataset, [train_size, test_size, support_size, remain_size])
    
    from torch.utils.data import Subset
    start_index = 3000
    end_index = int(0.2 * data.shape[0])
    indices = list(range(start_index, 3000+end_index))
    subset_data = Subset(all_dataset, indices)

    return train_data, test_data, support_data

def main(model_name, split_list, support_size, emb_dim=24):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    random_seed = 0

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # dataset_name_dict = {'Mistral-7B': ['hotpotqa']}
    # dataset_name_dict = {'meta-llama/Llama-2-7b-hf': ['hotpotqa']}
    dataset_name_dict = {'openai-community/gpt2-xl': ['nq']}
    # dataset_name_dict = {'Llama-2-13b': ['hotpotqa']}
    
    act_dim_dict = {'openai-community/gpt2-xl': 6400, 'meta-llama/Llama-2-7b-hf': 11008, 'Mistral-7B': 14336, 'vicuna-7b-v1.5': 11008,
                    'Llama3-8b': 14336, 'Llama-2-13b': 13824, 'vicuna-13b-v1.5': 13824, 'stablelm-tuned-alpha-7b': 24576}

    act_dim = act_dim_dict[model_name]
    dataset_name = dataset_name_dict[model_name]

    print("In distribution dataset: ", dataset_name)
    all_activation_values, all_label, all_source = load_data(dataset_name, model_name)
    print('all_activation_values:', all_activation_values.shape)
    all_data = all_activation_values.cpu().numpy()

    train_data, test_data, support_data = split_set(
        all_data, all_label, all_source, support_size, split_list)
    train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    support_loader = data.DataLoader(
        support_data, batch_size=64, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    with h5py.File('./features/{}/test_support_data.h5'.format(model_name), 'w') as f:
        f.create_dataset('test_data_features', data=test_data.dataset.features[test_data.indices])
        f.create_dataset('test_data_label', data=test_data.dataset.labels[test_data.indices])
        f.create_dataset('test_data_source', data=test_data.dataset.source[test_data.indices])
        f.create_dataset('support_data_features', data=support_data.dataset.features[support_data.indices])
        f.create_dataset('support_data_label', data=support_data.dataset.labels[support_data.indices])
        f.create_dataset('support_data_source', data=support_data.dataset.source[support_data.indices])

    # init model
    act_resnet_model = models.resnet18(
    pretrained=False, num_classes=emb_dim).cuda()
    act_resnet_model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()

    act_model = ActTriNet(act_resnet_model).cuda()
    
    # train the model
    print('act:')
    train_and_evaluate_model(act_model, train_loader, test_loader,
                            support_loader, './act_trinet.pth'.format(model_name), act_dim)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and Evaluate Neural Network')
    parser.add_argument('--model_name', type=str,
                        default='openai-community/gpt2-xl', help='Name of the model to use')
    parser.add_argument('--split_list', type=float, nargs='+',
                        default=[0.7, 0.2], help='List of splits for the dataset')
    parser.add_argument('--support_size', type=int,
                        default=100, help='Size of the support set') # support dataset

    args = parser.parse_args()
    main(args.model_name, args.split_list, args.support_size)

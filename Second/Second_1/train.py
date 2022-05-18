import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import resnet34
from ArabicCharactersDataset import ArabicCharactersDataset

def main(batch_size=32, lr=0.0001, epochs=3):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers every process".format(num_workers))

    data_root = os.path.abspath(os.path.join(os.getcwd(), "archive"))
    train_image_root = os.path.join(data_root, "Train Images 13440x32x32/train")
    test_image_root = os.path.join(data_root, "Test Images 3360x32x32/test")
    assert os.path.exists(train_image_root), "{} path does not exist".format(train_image_root)
    assert os.path.exists(test_image_root), "{} path does not exist".format(test_image_root)

    train_dataset = ArabicCharactersDataset(train_image_root)
    train_num = len(train_dataset)
    test_dataset = ArabicCharactersDataset(test_image_root)
    test_num = len(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("Using {} images for training, {} images for testing".format(train_num, test_num))

    model = resnet34(num_classes=28)
    # load pretrained weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./resnet34-333f7ec4.pth"
    # assert os.path.exists(model_weight_path), \
    #     "Pretrained weights file {} does not exist, you can downlaod " \
    #     "it from: https://download.pytorch.org/models/resnet34-333f7ec4.pth".format(model_weight_path)
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)

    # define loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    best_acc = 0.0
    save_path = './ResNet.pth'
    train_steps = len(train_dataloader)

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device).to(torch.float32))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss: {:.3f}".format(epoch+1, epochs, loss)

        model.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_dataloader)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = model(test_images.to(device).to(torch.float32))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

                test_bar.desc = "test epoch[{}/{}]".format(epoch+1, epochs)

        test_accuracy = acc/test_num
        print("[epoch %d] train_loss: %.3f test_accuracy: %.3f" % (epoch+1, running_loss/train_steps, test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), save_path)

    print("Finish Training")

if __name__  == "__main__":
    main()



"""
目的：输入数据集和epoch之后开始训练，训练五轮，然后把问题图片剔除出来
保存五个权重，五个log
"""
# train.py
import os

import shutil
import time

from PIL import Image

from tqdm import tqdm

from torchvision import transforms, datasets
import torch.optim as optim
from resnet import *
import json


class NegEntropy(object):
    """
    1、把原本的output里面的负数删除掉，变成趋近于0的一个数，然后去softmax，而不是直接softmax
    2、log以e为底probs的对数*probs
    3、batchsize=128，class=10，把128*10通过求和的方式变成128个数
    4、然后对128个数求平均，起了个名字叫负熵（富商）


    考虑一个极端的情况来理解负熵：
        十个类只有一个值是1，其他都是0
    """

    def __call__(self, outputs):
        outputs = outputs.clamp(min=1e-12)  # 把小于零的变成10的-12次方，趋近于0的一个数
        probs = torch.softmax(outputs, dim=1)  # 归一化指数函数，预测的概率为非负数、各种预测的概率之和为1
        return torch.mean(torch.sum(probs.log() * probs, dim=1))  # log是   ln（）-》就是log以e为底的对数   xlnx


def train2train_val(train_path, ratio=0.2):
    new_train_path = f"{os.path.dirname(train_path)}/train"
    new_val_path = f"{os.path.dirname(train_path)}/val"
    for cls_name in os.listdir(train_path):
        cls_image_list_path = os.path.join(train_path, cls_name)
        train_cls_image_list_path = os.path.join(new_train_path, cls_name)
        val_cls_image_list_path = os.path.join(new_val_path, cls_name)
        os.makedirs(train_cls_image_list_path, exist_ok=True)
        os.makedirs(val_cls_image_list_path, exist_ok=True)

        val_num = ratio * len(os.listdir(cls_image_list_path))
        count = 0
        for img_name in tqdm(os.listdir(cls_image_list_path)):
            count += 1
            img_path = os.path.join(cls_image_list_path, img_name)
            if count <= val_num:
                dst_image_path = os.path.join(val_cls_image_list_path, img_name)
            else:
                dst_image_path = os.path.join(train_cls_image_list_path, img_name)
            shutil.copy(img_path, dst_image_path)


def train(class_num, epochs, train_loader, valid_loader, val_dataset_num, test_loader, test_dataset_num, log_path,
          weights_save_path,
          pre_weights_path=None):
    # resnet18
    net = ResNet(BasicBlock, [2, 2, 2, 2],
                 num_classes=class_num)  ############################################################################
    if pre_weights_path:
        net.load_state_dict(torch.load(pre_weights_path))
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    # 优化器 这里用Adam
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # 训练参数保存路径

    # 训练过程中最高准确率
    best_acc = 0.0

    train_loss_record = []
    val_pre_record = []
    test_pre_record = []

    # 开始进行训练和测试，训练一轮，测试一轮
    # conf_penalty = NegEntropy()
    best_valid_epoch = 0
    save_path_1 = f'{weights_save_path}/normal.pth'
    for epoch in range(epochs):
        save_path_2 = f'{weights_save_path}/over_fitting.pth'
        # 上一个save_path_2
        # last_save_path_2 = f'{weights_save_path}/over_fitting_{epoch}epochs.pth'

        # train
        net.train()  # 训练过程中，使用之前定义网络中的dropout
        running_loss = 0.0

        # t1 = time.perf_counter()
        # 一个batch一次传播
        step = 0
        for step, data in enumerate(train_loader, start=0):
            # print(step)
            images, labels = data
            optimizer.zero_grad()
            _, outputs = net(images.to(device))
            # penalty = conf_penalty(outputs)
            # loss = loss_function(outputs, labels.to(device)) + penalty
            loss = loss_function(outputs, labels.to(device))
            # print(f"***{epoch+1}-{step + 1}:  loss is {loss}")
            # print(f"模型输出是：{outputs}, label是：{labels.to(device)},求出的loss是{loss}")
            # labels.to(device)意思是把数据copy一份到device上，这里的device是上面判断过的，如果有gpu就用gpu，如果没有就用cpu
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            # print(rate, step, len(train_loader))
            # 69
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\r epoch:{}      train loss: {:^3.0f}%[{}->{}]{:.3f}".format(epoch + 1, int(rate * 100), a, b, loss),
                  end="")
        train_loss_record.append(running_loss / (step + 1))
        # print(f'step:{step}')
        # print()
        # print(time.perf_counter() - t1)

        # validate 验证效果比较好的才会留下
        net.eval()  # 测试过程中不需要dropout，使用所有的神经元
        val_acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():  # 进一步加速和节省gpu空间，因为不需要计算和存储梯度
            for val_data in valid_loader:  # 每个batch每个batch的测试
                val_images, val_labels = val_data
                _, outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # list中存放着32个图片的预测结果
                # 将所有检测和标签对应的，保存起来
                val_acc += (predict_y == val_labels.to(device)).sum().item()  # 走完所有batch后，总的测试集正确的个数
            # map
            val_accurate = val_acc / val_dataset_num  # 正确的个数/总的个数
            # 验证集效果最好，且训练次数超过百分之三十
            if val_accurate > best_acc and epoch + 1 > int(epochs / 3):
                best_valid_epoch = epoch + 1
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path_1)
            val_pre_record.append(val_accurate)

        # test
        test_acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():  # 进一步加速和节省gpu空间，因为不需要计算和存储梯度
            for test_data in test_loader:  # 每个batch每个batch的测试
                test_images, test_labels = test_data
                _, outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # list中存放着32个图片的预测结果
                # 将所有检测和标签对应的，保存起来
                test_acc += (predict_y == test_labels.to(device)).sum().item()  # 走完所有batch后，总的测试集正确的个数
            # map
            test_accurate = test_acc / test_dataset_num  # 正确的个数/总的个数
            test_pre_record.append(test_accurate)

        # 保存第n个epochs的模型，然后删除n-1个epochs的模型
        torch.save(net.state_dict(), save_path_2)
        # if epoch != 0:
        #     os.remove(last_save_path_2)
        print(
            f"{epoch + 1} epoch: 训练集平均batch_loss{running_loss / (step + 1)}，验证集准确率{val_accurate}，测试集准确率{test_accurate}")
    # # 最后一个save_path_1
    # if os.path.exists(save_path_1):
    #     last_save_path_1 = f'{weights_save_path}/normal_{best_valid_epoch}epochs.pth'
    #     # 最后结束后会把normal重命名为valid效果最好的那个epoch
    #     os.rename(save_path_1, last_save_path_1)
    with open(log_path, 'w') as f:
        f.write(f"train_loss_record:{train_loss_record}\n")
        f.write(f'val_pre_record:{val_pre_record}\n')
        f.write(f'test_pre_record:{test_pre_record}\n')


def remove_bad(model, data_path, class_indict, dst_path):
    bad_image_path_list = []
    dst_dir_path_1 = f"{dst_path}/bad"
    dst_dir_path_2 = f"{dst_path}/train_src"
    os.makedirs(dst_dir_path_1, exist_ok=True)

    model.eval()
    num_classes = len(class_indict)
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 二分类的话是挑选置信度高于90而且判断错误的 （错误的前五分之一） 3： 13/15      2： 9/10
    confident_threshold_1 = round((4 * num_classes + 1) / (5 * num_classes), 3)
    # 二分类的话挑选预测概率在0.5到0.6之间的数据 （正确的后五分之一）             3:7/15   2:6/10
    confident_threshold_2 = round((num_classes + 4) / (5 * num_classes), 3)

    for cls_dir in os.listdir(data_path):
        cls_dir_path = os.path.join(data_path, cls_dir)
        cls_images_list = tqdm(os.listdir(cls_dir_path))
        os.makedirs(f"{dst_dir_path_2}/{cls_dir}", exist_ok=True)
        all_num = 0
        for image in cls_images_list:
            all_num += 1
            image_path = os.path.join(cls_dir_path, image)
            img = Image.open(image_path)

            img = data_transform(img)

            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # predict class
                _, output = model(img.to(device))
                output = torch.squeeze(output)
                predict_cla = torch.argmax(output).cpu().numpy()
                # 错的置信度还高
                if class_indict[str(predict_cla)] != cls_dir and output[predict_cla] > confident_threshold_1:
                    # print(
                    #     f"这张图标签为{cls_dir}，模型检测结果为{class_indict[str(predict_cla)]}，且置信度为{output[predict_cla]}>{confident_threshold_1}")
                    dst_path = f"{dst_dir_path_1}/{image}"
                    bad_image_path_list.append(image_path)
                elif output[predict_cla] < confident_threshold_2:
                    # print({
                    #     f"这张图的标签为{cls_dir}，置信度为{output[predict_cla]}<{confident_threshold_2}"})
                    dst_path = f"{dst_dir_path_1}/{image}"
                    bad_image_path_list.append(image_path)
                else:
                    dst_path = f"{dst_dir_path_2}/{cls_dir}/{image}"
                shutil.copy(image_path, dst_path)
    return bad_image_path_list


if __name__ == '__main__':
    # device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    dataset_path = r'E:\pycharmproject\cleantest\data\大面cls网(指纹脏污分开)_1029-2023-2-8 15-15-19_placed_well'
    project_name = '大面指纹脏污'
    batch_size = 32
    epochs = 100
    train_num = 5
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    src_train_path = f"{dataset_path}/train_src"
    # 将train_src变成train和val，用拷贝的方式，train_src不变
    train2train_val(src_train_path)

    # 数据转换
    testate_dataset = datasets.ImageFolder(root=dataset_path + "/test",
                                           transform=data_transform)
    test_dataset_num = len(testate_dataset)
    test_loader = torch.utils.data.DataLoader(testate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

    classes_list = testate_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    class_indict = dict((str(val), key) for key, val in classes_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    json_path = f'./runs/json/class_indices_{project_name}.json'
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)
    train_output_save_path = dataset_path
    print(f"训练即将开始...")
    time.sleep(3)
    # 五次挑选出问题图片
    bad_pic_path_and_bad_score = {}
    for train_idx in range(1, train_num + 1):
        weights_save_path = f"./runs/weights/{project_name}_{train_idx}"
        os.makedirs(weights_save_path, exist_ok=True)
        log_path = f'./runs/log/{project_name}_cleantest_{epochs}epochs_{train_idx}.txt'
        train_dataset = datasets.ImageFolder(root=dataset_path + "/train",
                                             transform=data_transform)
        train_dataset_num = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=0)

        validate_dataset = datasets.ImageFolder(root=dataset_path + "/val",
                                                transform=data_transform)
        val_dataset_num = len(validate_dataset)
        valid_loader = torch.utils.data.DataLoader(validate_dataset,
                                                   batch_size=batch_size, shuffle=False,
                                                   num_workers=0)
        # 训练一次
        train(len(class_indict), epochs, train_loader, valid_loader, val_dataset_num, test_loader, test_dataset_num,
              log_path,
              weights_save_path)
        print(f"第{train_idx}次训练结束，正在准备剔除问题数据...")
        time.sleep(3)
        # 训练生成的模型进行训练集清洗
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=len(class_indict))
        model.to(device)
        # load model weights
        model_weight_path = f'{weights_save_path}/normal.pth'
        model.load_state_dict(torch.load(model_weight_path))
        dataset_path = f"{train_output_save_path}/train_{train_idx}"
        bad_image_path_list = remove_bad(model, src_train_path, class_indict, dataset_path)
        print(bad_image_path_list)
        for image_path in bad_image_path_list:
            if image_path in bad_pic_path_and_bad_score:
                bad_pic_path_and_bad_score[image_path] += 1
            else:
                bad_pic_path_and_bad_score[image_path] = 1
        if train_idx == train_num:
            print(
                f"问题图片挑选实验结束，这是所有的问题图片路径和错误分数{bad_pic_path_and_bad_score}，\n准备进行最终的训练...")
            time.sleep(3)
            break
        print(f"第{train_idx}次训练的问题图片已经剔除，正在准备划分下次训练的数据集...")
        time.sleep(3)
        # 将新的划分成train和val，然后把训练地址更新
        train_path = f"{dataset_path}/train_src"
        train2train_val(train_path)
        print(f"下次训练的数据集已经划分好，正在准备进行下一次训练...")
        time.sleep(3)

    # 整理出实验的干净数据
    print("正在根据整理出干净的数据集")
    picked_bad_image_path_list = [key for key, value in bad_pic_path_and_bad_score.items() if value >= 3]
    print(f"这些是实验整理出的问题图片{picked_bad_image_path_list}")

    dst_dataset_path = f"{train_output_save_path}/train_clean"
    dst_bad_path = f"{dst_dataset_path}/bad"
    dst_train_src_path = f"{dst_dataset_path}/train_src"
    os.makedirs(dst_bad_path, exist_ok=True)
    for cls_name in os.listdir(src_train_path):
        cls_path = os.path.join(src_train_path, cls_name)
        dst_cls_path = os.path.join(dst_train_src_path, cls_name)
        os.makedirs(dst_cls_path, exist_ok=True)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            if img_path in picked_bad_image_path_list:
                dst_img_path = os.path.join(dst_bad_path, img_name)
            else:
                dst_img_path = os.path.join(dst_cls_path, img_name)
            shutil.copy(img_path, dst_img_path)
    # 将干净数据进行最后一次训练
    train2train_val(dst_train_src_path)
    train_dataset = datasets.ImageFolder(root=dst_dataset_path + "/train",
                                         transform=data_transform)
    train_dataset_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=dst_dataset_path + "/val",
                                            transform=data_transform)
    val_dataset_num = len(validate_dataset)
    valid_loader = torch.utils.data.DataLoader(validate_dataset,
                                               batch_size=batch_size, shuffle=False,
                                               num_workers=0)
    weights_save_path = f"./runs/weights/{project_name}_clean"
    os.makedirs(weights_save_path, exist_ok=True)
    log_path = f'./runs/log/{project_name}_cleantest_{epochs}epochs_clean.txt'
    train(len(class_indict), epochs, train_loader, valid_loader, val_dataset_num, test_loader, test_dataset_num,
          log_path,
          weights_save_path)

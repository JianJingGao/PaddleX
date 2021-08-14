from paddlex.det import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.ResizeByShort(),
    transforms.RandomDistort(),
    transforms.Normalize(),

])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.ResizeByShort(),
    transforms.RandomDistort(),
    transforms.Normalize(),
])

import paddlex as pdx

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/barricade',
    file_list='objDataset/barricade/train_list.txt',
    label_list='objDataset/barricade/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/barricade',
    file_list='objDataset/barricade/val_list.txt',
    label_list='objDataset/barricade/labels.txt',
    transforms=eval_transforms)


# 此处需要补充目标检测模型代码
model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='MobileNetV1')

# 此处需要补充模型训练参数
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/yolov3_mobilenetv1')
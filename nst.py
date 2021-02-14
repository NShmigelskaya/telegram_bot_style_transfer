
from PIL import Image
import io


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import asyncio

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import os

#path_user_photos = os.getcwd() + os.sep + "User_photos"
#PATH = os.path.join(path_user_photos, str(user_id))
#file_names = ['1.jpg', '2.jpg', '3.jpg']

imsize = 128  # задаём размер изображения

loader = transforms.Compose([
    transforms.Resize(imsize),  # нормируем размер изображения
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # превращаем в удобный формат

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

#style_img1 = image_loader(os.path.join(path_user_photos, str(user_id), str(file_names[0])))
#content_img = image_loader(os.path.join(path_user_photos, str(user_id), str(file_names[2])))
#style_img2 = image_loader(os.path.join(path_user_photos, str(user_id), str(file_names[1])))

unloader = transforms.ToPILImage() # тензор в кратинку
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class ContentLoss(nn.Module):
    
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.l1_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.l1_loss(input, self.target)
        return input

def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()  # batch_size = 1
    features = input.view(batch_size * h, w * f_map_num)
    G = torch.mm(features, features.t())  # задаём матрицу Грама
    return G.div(batch_size * h * w * f_map_num)  # нормализуем значения матрицы Грама

class StyleLoss(nn.Module):
    def __init__(self, target_feature1, target_feature2):
        super(StyleLoss, self).__init__()
        # задаём 2 маски, первая - нижнетреугольная, вторая - верхнетреугольная
        self.mask1 = torch.tril(torch.ones(target_feature1.size()[2], target_feature1.size()[3])).to(device)
        self.mask1 = torch.cat(target_feature1.size()[1] * [self.mask1.unsqueeze(0)]).unsqueeze(0).detach().to(
            device)
        self.mask2 = (torch.ones(target_feature2.size()[2], target_feature2.size()[3]) - torch.tril(
        torch.ones(target_feature2.size()[2], target_feature2.size()[3]))).to(device)
        self.mask2 = torch.cat(target_feature2.size()[1] * [self.mask2.unsqueeze(0)]).unsqueeze(0).detach().to(
            device)
        # рассчитываем таргеты для кадой маски по отдельности,затем суммируем
        self.target1 = gram_matrix(target_feature1 * self.mask1).detach()
        self.target2 = gram_matrix(target_feature2 * self.mask2).detach()
        self.target = self.target1 + self.target2
        self.loss = F.l1_loss(self.target, self.target)

    def forward(self, input):
        # создаём матрицы Грама для каждой маски, затем суммируем
        G1 = gram_matrix(input * self.mask1)
        G2 = gram_matrix(input * self.mask2)
        G = G1 + G2
        self.loss = F.l1_loss(G1, self.target1) + F.l1_loss(G2, self.target2)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


cnn = models.vgg19(pretrained=True).features.to(device).eval()


class StyleTransfer:

    def __init__(self, style_img1, style_img2, content_img, cnn):
        self.style_img1 = style_img1
        self.style_img2 = style_img2
        self.content_img = content_img
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.content_layers = ['conv_10']
        self.style_layers = ['conv_1', 'conv_3', 'conv_5', 'conv_7', 'conv_8', 'conv_10']
        self.input_img = content_img.clone()
        self.num_steps = 10
        self.style_weight = 100000
        self.content_weight = 15
        self.cnn = cnn #

    def change_layers(self):
        for i, layer in enumerate(self.cnn):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def get_style_model_and_losses(self):
        cnn = copy.deepcopy(self.cnn)
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):  # замена nn.MaxPool2d на nn.AvgPool2d
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # посчитаем style_loss от каждого из стилей
                target_feature1 = model(self.style_img1).detach()
                target_feature2 = model(self.style_img2).detach()
                style_loss = StyleLoss(target_feature1, target_feature2)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # выбрасываем все уровни после последенего styel_loss или content_loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self):
        # добавляем в оптимизатор weight_decay=0.1, так регуляризация будет слабее менять пиксели и перенос стиля на контент будет более равномерным
        #optimizer = optim.Adam([self.input_img.requires_grad_()], lr=0.01, weight_decay=0.1)
        optimizer = optim.LBFGS([self.input_img.requires_grad_()], max_iter=1)
        return optimizer

    async def run_style_transfer(self, style_weight=100000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses()
        optimizer = self.get_input_optimizer()

        print('Optimizing..')
        run = [0]
        while run[0] <= self.num_steps:

            def closure():
                # нужно для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                self.input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(self.input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # взвешивание ошибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)
            await asyncio.sleep(0.1)

        self.input_img.data.clamp_(0, 1)

        return self.input_img


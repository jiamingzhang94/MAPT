import torch


class BaseAttack:
    """对抗攻击的基类"""

    def __init__(self, eps=2. / 255, steps=10, alpha=None):
        self.eps = eps
        self.steps = steps
        self.alpha = alpha if alpha is not None else eps / steps * 1.5

    def generate(self, model, images, labels, normalize=None):
        """生成对抗样本的接口方法"""
        raise NotImplementedError


class PGDAttack(BaseAttack):
    """PGD攻击实现"""

    def generate(self, model, images, labels, normalize=None):
        if normalize is not None:
            images = normalize(images)

        delta = torch.zeros_like(images).cuda()
        delta.requires_grad = True

        for _ in range(self.steps):
            if delta.grad is not None:
                delta.grad.zero_()

            loss = model(images + delta, labels)
            loss.backward()

            grad = delta.grad.detach()
            delta.data = delta + self.alpha * torch.sign(grad)
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)

        delta = delta.detach()
        return torch.clamp(images + delta, 0, 1)


# 可以添加更多攻击方法
class FGSMAttack(BaseAttack):
    """FGSM攻击实现"""

    def generate(self, model, images, labels, normalize=None):
        if normalize is not None:
            images = normalize(images)

        images.requires_grad = True
        loss = model(images, labels)
        loss.backward()

        delta = self.eps * torch.sign(images.grad.detach())
        return torch.clamp(images + delta, 0, 1)

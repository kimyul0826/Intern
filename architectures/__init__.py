from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50
from .densenet import DenseNet40, DenseNet100, DenseNetBC100
from .fractalnet import FractalNet20, FractalNet40

models = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "preact_resnet18": PreActResNet18,
    "preact_resnet34": PreActResNet34,
    "preact_resnet50": PreActResNet50,
    "densenet40": DenseNet40,
    "densenet100": DenseNet100,
    "densenet_bc100": DenseNetBC100,
    "fractalnet20": FractalNet20,
    "fractalnet40": FractalNet40,
}

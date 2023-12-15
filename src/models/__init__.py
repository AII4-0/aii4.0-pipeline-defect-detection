import yaml

from src.models.cnn_defect_detector import CNNDefectDetector
from src.models.resnet50_defect_detector import ResNet50DefectDetector

params = yaml.safe_load(open("params.yaml"))
prepare_params = params["prepare"]
train_params = params["train"]
model_name = train_params["model_name"]

model_registry = {}

if model_name == "resnet50_defect_detector_v1":
    model_registry["resnet50_defect_detector_v1"] = ResNet50DefectDetector(
        img_shape=prepare_params["img_shape"],
        hiddens=train_params["hiddens"],
        dropout=train_params["dropout"],
        lr=train_params["lr"],
        num_classes=train_params["num_classes"],
    )
elif model_name == "cnn_defect_detector_v1":
    model_registry["cnn_defect_detector_v1"] = CNNDefectDetector(
        img_shape=prepare_params["img_shape"],
        convs=train_params["convs"],
        hiddens=train_params["hiddens"],
        dropout=train_params["dropout"],
        lr=train_params["lr"],
        num_classes=train_params["num_classes"],
    )

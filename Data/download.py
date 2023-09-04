from roboflow import Roboflow


rf = Roboflow(api_key="XErNFc6ZaONvtcWDdusH")
project = rf.workspace("fseg-r6czn").project("fseg-food-detection")
dataset = project.version(1).download("coco")

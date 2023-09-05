from segment_anything import sam_model_registry, SamPredictor



def downloadSAMModel():
    return None


def LoadSAMPredictor(sam_checkpoint, model_type, device='cuda'):
    """
    Load a Segment Anything Model (SAM) predictor model for semantic segmentation.

    Parameters:
    - sam_checkpoint (str): The path to the checkpoint file containing the SAM model's weights and configuration.
    - model_type (str): The SAM model type to use. It should be a key that corresponds to a model in the 'sam_model_registry'.
    - device (str, optional): The device to run the model on, either 'cuda' (GPU) or 'cpu' (CPU). Default is 'cuda'.

    Returns:
    - predictor (SamPredictor): An instance of the SAM predictor configured with the specified model type and loaded weights.
    """
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return predictor


def GenerateMask(predictor, input_box):
    """
    Generate segmentation masks for a given input bounding box using a SAM predictor.

    Parameters:
    - predictor (SamPredictor): An instance of the SAM predictor loaded with a model.
    - input_box (list or numpy array): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].

    Returns:
    - masks (numpy array): A segmentation mask representing the predicted object(s) within the input bounding box.
    """
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    return masks




import pathlib
import xml.etree.ElementTree as ET

from torchvision.datasets.utils import download_url
from urllib.error import URLError
import click
import numpy as np
import os
import sys
import tifffile as tiff
import torch
from rich import traceback
from stainsegmy.model.unet_instance import Unet, UneXt, ContextUnet
from stainsegmy.patch_extractor.patch_extractor import PatchExtractor, MaskStitcher
from stainsegmy.model.utils import weights_init


WD = os.path.dirname(__file__)


__version__ = "0.1.2"


@click.command()
@click.option('-i', '--input', type=str, help='Path to data file to predict.')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-o', '--output', default="", type=str, help='Folder path to write the output to')
@click.option('-s/-ns', '--sanitize/--no-sanitize', type=bool, default=False, help='Whether to remove model after ''prediction or not.')
@click.option('-m', '--model', type=str, default="dummy.ckpt", help="Path to model")
@click.option('--architecture', type=str, default="U-Net", help="U-Net or U-Next or CU-Net")
@click.option('--version', is_flag=True, help='Show version and exit')


def main(input: str, cuda: bool, output: str, sanitize: bool, model: str, architecture: str, version: bool):
    """
    Main function to run the prediction pipeline. It loads the model, reads the input image, extracts patches, 
    predicts segmentation masks for each patch, stitches the masks together, and saves the final segmentation mask 
    as an OME-TIFF file.

    parameters:
    - input (str): Path to the input image file to predict on.
    - cuda (bool): Whether to use CUDA for prediction.
    - output (str): Folder path to write the output segmentation mask to.
    - sanitize (bool): Whether to remove the model file after prediction.
    - model (str): Path to the PyTorch model checkpoint to use for prediction.
    - architecture (str): The architecture of the model (e.g., "U-Net", "U-NeXt", "CU-Net").
    - version (bool): Whether to print the version and exit.

    returns:
    - None
    """

    if version:
        print(__version__)
        return
    
    if input is None:
        raise click.UsageError("Missing required option '--input'")
    if output is None:
        raise click.UsageError("Missing required option '--output'")

    win_size = (256, 256)
    step_size = (206, 206)

    model = get_pytorch_model(model, sanitize, architecture)

    if cuda:
        if torch.cuda.is_available():
            print("Using GPU..........")
            model.cuda()
        else:
            print("CUDA requested but not available. Falling back to CPU..........")
    else:
        print("Using CPU..........")

    image = read_data_to_predict(input)

    if image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0)) # ensure image is in (H, W, C) format

    im_h, im_w = image.shape[:2]
    patch_extractor = PatchExtractor(image, win_size=win_size[0], step_size=step_size[0])
    stitch_mask = MaskStitcher(im_h, im_w, win_size[0], step_size[0])

    h_last = patch_extractor.extract_infos(im_h, win_size[0], step_size[0])
    w_last = patch_extractor.extract_infos(im_w, win_size[1], step_size[1])

    # function to determine if the current patch is an edge patch and to which edge(s) it belongs to
    def is_edge_patch(row, col, h_last, w_last):
        end_x, end_y = False, False

        if row == 0 and row + win_size[0] > h_last:
            end_y = "both"
        elif row == 0:
            end_y = "top"
        elif row + win_size[0] > h_last:
            end_y = "bottom"

        if col == 0 and col + win_size[1] > w_last:
            end_x = "both"
        elif col == 0:
            end_x = "left"
        elif col + win_size[1] > w_last:
            end_x = "right"

        return end_x, end_y
    

    for row_idx, row in enumerate(range(0, h_last+1, step_size[0])):
        for col_idx, col in enumerate(range(0, w_last+1, step_size[1])):
            end_x, end_y = is_edge_patch(row, col, h_last, w_last)

            patch = patch_extractor.get_patch((row, col))

            result = predict(patch.transpose(2, 0, 1), model) # model expects (C, H, W) format
            result_mask = mask_binning(result[0, :, :, :]) 
            stitch_mask.stitch(result_mask, (row, col), end_x=end_x, end_y=end_y) 

            #print(f"Prediction completed for patch at row {row_idx+1}, col {col_idx+1}")
    
    print("Prediction completed for all patches")

    mask = stitch_mask.get_mask()

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, "Segmentation_mask.ome.tif")

    write_ome_out(mask, out_path=output_path, image_path=input)
    print(f"Segmentation mask saved to {output_path}")



def read_data_to_predict(path_to_data_to_predict: str):
    """
    Read the input image data for prediction. The image must be in RGB format (3 channels) and 2D (C, Y, X).

    Parameters:
    - path_to_data_to_predict (str): Path to the input image file to predict on

    Returns:
    - image (ndarray): The input image data read from the specified path, in RGB format and 2D shape
    """

    image = tiff.imread(path_to_data_to_predict)

    if not image.shape[0] >= 3 or not len(image.shape) == 3:
        raise ValueError("Image must be RGB (3 channels) and 2D (C, Y, X)")

    return image[:, :, :]



def _check_exists(filepath) -> bool:
    """
    Check if the model file exists at the specified filepath.

    Parameters:
    - filepath (str): The path to the model file to check for existence

    Returns:
    - bool: True if the file exists, False otherwise
    """
    return os.path.exists(filepath)



def download(architecture) -> None:
    """
    Download the model if it doesn't exist in processed_folder already.

    Parameters:
    - architecture (str): The architecture of the model to download (e.g., "U-Net", "U-Next", "CU-Net")

    Returns:
    - None
    """

    mirrors = [
        'https://zenodo.org/record/',
    ]
    if architecture == "U-Net":
        resources = [
            ("U_Net.ckpt", "19631105/files/U_Net.ckpt", "4644cb6d10ebfe2e672e82e6e45c0872"),
        ]
    elif architecture == "U-NeXt":
        resources = [
            ("U_NeXt.ckpt", "19631105/files/U_NeXt.ckpt", "92157b2f8b1b5f20ca3713355a1ccdc4"),
        ]
    elif architecture == "CU-Net":
        resources = [
            ("CU_NET.ckpt", "7884684/files/CU_NET.ckpt", "9090252a639c39c9f9509df7e1ce311c"),
        ]
    else:
        raise IOError("No architecture found")
    # download files
    for filename, uniqueID, md5 in resources:
        for mirror in mirrors:
            url = "{}{}".format(mirror, uniqueID)
            try:
                print("Downloading {}".format(url))
                download_url(
                    url, root=str(pathlib.Path("models/" + architecture).parent.absolute()),
                    filename=filename,
                    md5=md5
                )
            except URLError as error:
                print(
                    "Failed to download (trying next):\n{}".format(error)
                )
                continue
            finally:
                print()
            break
        else:
            raise RuntimeError("Error downloading {}".format(filename))



def get_pixel_info_ome_xml(ome_xml):
    """
    Get pixel size from OME-XML metadata.

    Parameters:
    - ome_xml (str): OME-XML metadata as a string

    Returns:
    - px (float): Physical size of a pixel in X direction (µm)
    - py (float): Physical size of a pixel in Y direction (µm)
    """

    root = ET.fromstring(ome_xml)
    pixels = root.find(".//{*}Pixels")   

    px = pixels.get("PhysicalSizeX")
    py = pixels.get("PhysicalSizeY")

    px = float(px) if px is not None else None
    py = float(py) if py is not None else None

    return px, py



def get_pixel_size_ome_tiff(file_path):
    """
    Get pixel size from an OME-TIFF file.

    Parameters:
    - file_path (str): Path to the OME-TIFF file

    Returns:
    - px (float): Physical size of a pixel in X direction (µm)
    - py (float): Physical size of a pixel in Y direction (µm)
    """
    
    with tiff.TiffFile(file_path) as tif:
        try:
            ome = tif.ome_metadata
            px, py = get_pixel_info_ome_xml(ome)
        except Exception as e:
            px, py = None, None

        return px, py



def write_ome_out(mask, out_path, physical_size_x=None, physical_size_y=None, image_path=None) -> None:
    """
    Save a segmentation mask as an OME-TIFF file with class labels.

    Parameters:
    - mask (ndarray): Segmentation mask to save (2D)
    - out_path (str): Output file path
    - physical_size_x (float or None): Pixel size in X (µm)
    - physical_size_y (float or None): Pixel size in Y (µm)
    - image_path (str or None): Path to the original image file to extract pixel size from if not provided

    Returns:
    - None
    """

    # add channel axis
    if mask.ndim == 2:
        mask_to_save = mask[np.newaxis, :, :]  # shape (1, Y, X)
    else:
        mask_to_save = mask

    # convert mask to integer data type
    max_label = mask.max()
    if max_label <= 255:
        mask = mask_to_save.astype(np.uint8)
    else:
        mask = mask_to_save.astype(np.uint16)

    if image_path is not None:
        physical_size_x, physical_size_y = get_pixel_size_ome_tiff(image_path)    
    
    with tiff.TiffWriter(out_path, ome=True, bigtiff=True) as tif:
        metadata = {
            "axes": "CYX",
            "PhysicalSizeX": physical_size_x,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": physical_size_y,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["Segmentation Mask"]},
        }

        tif.write(
            mask,
            metadata=metadata,
            compression=None
        )



def mask_binning(classification: torch.Tensor):
    """
    Convert the model output tensor to an array of class labels by taking the argmax across the channel dimension.

    Parameters:
    - classification (torch.Tensor): The output tensor from the model

    Returns:
    - classification (ndarray): an array of class labels
    """
    classification = classification.detach().cpu().numpy()
    classification = np.argmax(classification, axis=0)

    return classification



def get_pytorch_model(path_to_pytorch_model: str, sanitize: bool, architecture: str):
    """
    Load the PyTorch model from the specified path. If the model file does not exist, it will be downloaded based 
    on the specified architecture.

    parameters:
    - path_to_pytorch_model (str): Path to the PyTorch model checkpoint to load
    - sanitize (bool): Whether to remove the model file after loading
    - architecture (str): The architecture of the model (e.g., "U-Net", "U-NeXt", "CU-Net")       

    returns:
    - model (torch.nn.Module): The loaded PyTorch model ready for prediction
    """

    if not _check_exists(path_to_pytorch_model):
        print("Model not found at {}.".format(path_to_pytorch_model)) if path_to_pytorch_model != "dummy.ckpt" else None

        if architecture in ["U-Net", "U-NeXt", "CU-Net"]:
            download(architecture)

            if architecture == "U-Net":
                model = Unet(hparams={}, input_channels=3, num_classes=7, flat_weights=False, dropout_val=True)
                model.apply(weights_init)
                state_dict = torch.load("models/U_Net.ckpt", map_location="cpu")
                path_to_pytorch_model = "models/U_Net.ckpt"
                #print("U-Net model loaded from zenodo")

            elif architecture == "U-NeXt":
                model = UneXt(hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
                model.apply(weights_init)
                state_dict = torch.load("models/U_NeXt.ckpt", map_location="cpu")
                path_to_pytorch_model = "models/U_NeXt.ckpt"
                #print("U-NeXt model loaded from zenodo")
                
            elif architecture == "CU-Net":
                model = ContextUnet(hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
                model.apply(weights_init)
                state_dict = torch.load("models/CU_NET.ckpt", map_location="cpu")
                path_to_pytorch_model = "models/CU_NET.ckpt"
                #print("CU-Net model loaded from zenodo")
        else:
            raise KeyError("Please provide a valid architecture or a path to the model")
        
    else:
        if architecture == "U-Net":
            model = Unet(hparams={}, input_channels=3, num_classes=7, flat_weights=False, dropout_val=True)
            print("U-Net model loaded from {}".format(path_to_pytorch_model))
            model.apply(weights_init)
            state_dict = torch.load(path_to_pytorch_model, map_location="cpu")

        elif architecture == "U-NeXt":
            model = UneXt(hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
            print("U-NeXt model loaded from {}".format(path_to_pytorch_model))
            model.apply(weights_init)
            state_dict = torch.load(path_to_pytorch_model, map_location="cpu")
    
        elif architecture == "CU-Net":
            model = ContextUnet(hparams={}, input_channels=3, num_classes=7, flat_weights=True, dropout_val=True)
            print("CU-Net model loaded from {}".format(path_to_pytorch_model))
            model.apply(weights_init)
            state_dict = torch.load(path_to_pytorch_model, map_location="cpu")
        else:
            raise KeyError("Please provide the architecture name [U-Net, U-NeXt, CU-Net]")
        
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model.eval()

    if sanitize:
        os.remove(path_to_pytorch_model)

    return model



def predict(img, model):
    """
    Predict the segmentation mask for the given input image using the provided model. 
    The input image is expected to be in the format (C, H, W) and will be converted to a PyTorch tensor before prediction.

    Parameters:
    - img (ndarray): The input image data to predict on, expected shape (C, H, W)
    - model (torch.nn.Module): The PyTorch model to use for prediction

    Returns:
    - prediction (torch.Tensor): The predicted segmentation mask output from the model
    """
    model.eval()

    img = np.asarray(img, dtype=np.float32)
    img_tensor = torch.from_numpy(img).float()

    device = next(model.parameters()).device  # get model device
    img_tensor = img_tensor.to(device)
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        prediction = model(img_tensor)

    return prediction



if __name__ == "__main__":
    traceback.install()
    sys.exit(main())

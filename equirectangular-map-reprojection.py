import cv2
import torch
from math import pi
import numpy as np

def resize_keep_aspect_ratio(image, target_width=None, target_height=None, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image while keeping its aspect ratio.

    Args:
        image (numpy.ndarray): Input image.
        target_width (int, optional): Target width for resizing. Defaults to None.
        target_height (int, optional): Target height for resizing. Defaults to None.
        interpolation (int, optional): Interpolation method. Defaults to cv2.INTER_LINEAR.

    Returns:
        numpy.ndarray: Resized image.
    """
    h, w = image.shape[:2]

    if target_width is None and target_height is None:
        raise ValueError("Either target_width or target_height must be specified.")

    if target_width is not None:
        # Calculate new height to keep aspect ratio
        scale = target_width / w
        new_width = target_width
        new_height = int(h * scale)
    else:
        # Calculate new width to keep aspect ratio
        scale = target_height / h
        new_width = int(w * scale)
        new_height = target_height

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    return resized_image

class SphericalMapper(torch.nn.Module):

    def __init__(self, width, height, offset):
        super(SphericalMapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.offset = offset
        self.width = width
        self.height = height
        #self.rotation_matrix =
        self.recalculate_rotation_matrix() # Rotation matrix (A)

    def forward(self):
        """
        Perform spherical mapping for the entire image in a vectorized manner.

        Returns:
            tuple: (mapx, mapy) Tensors for the entire image.
        """
        # Generate grid of row and column indices
        cols, rows = torch.meshgrid(
            torch.arange(self.width, dtype=torch.float32),
            torch.arange(self.height, dtype=torch.float32),
            indexing="xy"
        )

        # Spherical coordinates (theta, phi)
        theta = 2.0 * pi * (self.width / 2 - cols) / self.width
        phi = pi * (self.height / 2 - rows) / self.height

        # Cartesian coordinates
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.cos(phi) * torch.cos(theta)
        z = torch.sin(phi)

        # Stack into a 3D coordinate tensor
        coords = torch.stack([x, y, z], dim=-1).to(self.rotation_matrix.device)

        # Rotate coordinates
        rotated_coords = torch.einsum("ij,hwi->hwj", self.rotation_matrix, coords)

        # Final spherical coordinates
        final_theta = torch.atan2(rotated_coords[..., 1], rotated_coords[..., 0])
        final_phi = torch.atan2(
            torch.sqrt(rotated_coords[..., 0]**2 + rotated_coords[..., 1]**2),
            rotated_coords[..., 2],
        )

        # Map to 2D UV coordinates
        mapx = self.width * final_theta / (2 * pi)
        mapy = final_phi * self.height / pi

        # Wrap around coordinates
        mapx = (mapx + self.width) % self.width
        mapy = (mapy + self.height) % self.height

        return mapx, mapy

    def change_offset(self, offset):
        self.offset = offset
        self.recalculate_rotation_matrix()

    def change_image_size(self, width, height):
        self.width = width
        self.height = height
        self.recalculate_rotation_matrix()

    def recalculate_rotation_matrix(self):
        w = self.width
        h = self.height
        offset = self.offset
        offset_x, offset_y = offset
        pianyi_theta = 2.0 * pi * (w / 2 - offset_x) / w - pi / 2.0
        pianyi_phi = pi * offset_y / h

        # Rotation matrix A
        A = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
        A[0, 0] = torch.cos(torch.tensor(pianyi_theta))
        A[0, 1] = -torch.sin(torch.tensor(pianyi_theta)) * torch.cos(torch.tensor(pianyi_phi))
        A[0, 2] = torch.sin(torch.tensor(pianyi_theta)) * torch.sin(torch.tensor(pianyi_phi))
        A[1, 0] = torch.sin(torch.tensor(pianyi_theta))
        A[1, 1] = torch.cos(torch.tensor(pianyi_theta)) * torch.cos(torch.tensor(pianyi_phi))
        A[1, 2] = -torch.cos(torch.tensor(pianyi_theta)) * torch.sin(torch.tensor(pianyi_phi))
        A[2, 0] = 0.0
        A[2, 1] = torch.sin(torch.tensor(pianyi_phi))
        A[2, 2] = torch.cos(torch.tensor(pianyi_phi))
        self.rotation_matrix = A  # Rotation matrix (A)




def spherical_mapping(src, mapper=None):
     # Initialize the SphericalMapper module
       # Compute mapx and mapy
    mapx, mapy = mapper()

    # Convert mapx and mapy to NumPy arrays for OpenCV remapping
    mapx_np = mapx.cpu().numpy().astype(np.float32)
    mapy_np = mapy.cpu().numpy().astype(np.float32)

    # Perform remapping
    remapped = cv2.remap(src, mapx_np, mapy_np, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


    return remapped

if __name__=="__main__":
#for i in range(36):
    image_path = "extracted_frame.jpg"
    for j in range(36):


        # Check for CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load the input image
        src = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if src is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        h, w, c = src.shape
        offset = (0, j*-100)
        # Offset calculations
        #if mapper is None:
        mapper = SphericalMapper(width=w, height=h, offset=offset).to(device)


        spherical_mapping(src, mapper)

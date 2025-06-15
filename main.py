import PyOpenColorIO as ocio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import av


def img_show(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_axis_off()
    plt.show()


def img_save(filepath: str, img: np.array):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, bgr)


def img_show_and_save(filepath: str, img: np.array):
    img_show(img)
    img_save(filepath, (img * 255).astype(np.uint8))

def load_frame(filepath: str):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError("Cannot read video")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def apply_cube_lut_with_ocio(img: np.ndarray, lut_path: str) -> np.ndarray:
    # Ensure float32 in [0,1]
    orig_dtype = img.dtype
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Create a minimal OCIO config and FileTransform
    config = ocio.Config.CreateRaw()
    ft = ocio.FileTransform()
    ft.setSrc(lut_path)
    ft.setInterpolation(ocio.INTERP_LINEAR)

    processor = config.getProcessor(ft)
    cpu_proc = processor.getDefaultCPUProcessor()

    # Flatten to Nx3
    flat = arr.reshape(-1, 3)
    # Apply in place; applyRGB returns None for NumPy buffers
    cpu_proc.applyRGB(flat)
    # Now flat contains the transformed pixels
    out = flat.reshape(arr.shape)

    # Convert back to original dtype/range if needed
    if np.issubdtype(orig_dtype, np.integer):
        out = np.clip(out, 0.0, 1.0)
        out = (out * 255.0).round().astype(orig_dtype)
    return out


def rec709_to_linear(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        raise ValueError("Image must be composed of normalized floats [0, 1]")

    mask = img < 0.081
    linear = np.empty_like(img)
    linear[mask] = img[mask] / 4.5
    linear[~mask] = ((img[~mask] + 0.099) / 1.099) ** (1.0 / 0.45)
    return linear


def exponential_blur(img: np.ndarray, sigma: float, kernel_size=None) -> np.ndarray:
    """
    img: H×W×C or H×W NumPy array as a float
    sigma: controls falloff. Larger sigma → slower decay → heavier blur.
    kernel_size: kernel radius in pixels; if None, use something like int(3*sigma).
    """
    if kernel_size is None:
        kernel_size = int(3 * sigma)

    # compute a grid of distances from center
    y, x = np.ogrid[-kernel_size:kernel_size + 1, -kernel_size:kernel_size + 1]
    dist = np.sqrt(x*x + y*y)
    # exponential falloff: w(r) = exp(-r / sigma)
    kernel = np.exp(-dist / sigma)
    kernel /= kernel.sum()

    # apply filter; use same depth

    # If a float image convert to uint8
    img = (img * 255).astype(np.uint8)

    blurred = cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)

    # Convert back to float
    blurred = (blurred / 255).astype(np.float32)

    return blurred


def halation(img: np.ndarray) -> np.ndarray:
    # This reddit thread describes how he applied halation to his images, and it looked pretty good
    # https://www.reddit.com/r/colorists/comments/17mwxmv/my_take_on_simulating_halation/
    # I do a combination of what the reddit thread suggests and what bloom algorithms in computer graphics use

    # 0. Threshold - ideally this would be better with an HDR image, but we just have to work with what we got
    threshold = 0.9
    thresholded = img.copy()
    thresholded[thresholded < threshold] = 0

    # 1. Create a blurred version of the original image (using an exponential falloff kernel)
    blurred = exponential_blur(thresholded, 20)

    img_show(blurred)

    # 2. Add a small amount of the blurred image back to the original, primarily in the red channel and a bit in the green
    blur_redshift = np.array([1, 0.5, 0.25], dtype=np.float32)
    halation_strength = 0.25
    with_halation = img + blurred * blur_redshift * halation_strength
    with_halation[with_halation > 1] = 1  # Clamp values since the addition may go above 1

    return with_halation


def denoise(img: np.ndarray, blur=True) -> np.ndarray:
    rgb8 = (img * 255.0).astype(np.uint8)
    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)

    denoised = np.zeros(img.shape, dtype=np.uint8)
    cv2.fastNlMeansDenoisingColored(bgr8, denoised, 4, 7)
    if blur:
        # This helps since we're trying to get a not-sharp looking image. This also should remove a bit of noise,
        #  but it's secondary to the denoiser
        denoised = cv2.GaussianBlur(denoised, ksize=(0, 0), sigmaX=0.5, sigmaY=0.5)
    denoised_rgb8 = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    return (denoised_rgb8 / 255.0).astype(np.float32)


def main():
    frame = load_frame("hammock_landscape.MOV")
    frame = frame.astype(np.float32) / 255.0
    img_show_and_save("out/original.png", frame)

    # Apply LUT
    lut_filepath = "apple_log_to_fujifilm_3513di_d65_rec709_g2-4.cube"

    frame_lut_rec709 = apply_cube_lut_with_ocio(frame, lut_filepath)
    img_show_and_save("out/lut.png", frame_lut_rec709)

    # Convert to linear space
    frame_linear = rec709_to_linear(frame_lut_rec709)
    img_show_and_save("out/linear.jpg", frame_linear)

    # Apply halation
    frame_halation = halation(frame_linear)
    img_show_and_save("out/halation.jpg", frame_halation)

    # Film grain
    # a) Denoise the image (since we don't want digital camera noise + film grain)
    #      This also makes our image look softer, which is good because film is known to be soft and digital cameras are
    #      known to be sharp.
    denoised = denoise(frame_halation)
    img_show(denoised)
    img_show_and_save("out/denoised.png", denoised)


if __name__ == "__main__":
    main()

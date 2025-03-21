def load_image(image_path):
    """Load an image from the specified file path."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def save_image(image, save_path):
    """Save an image to the specified file path."""
    import cv2
    cv2.imwrite(save_path, image)

def preprocess_image(image, target_size=(224, 224)):
    """Resize and normalize the image for model input."""
    import cv2
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def draw_rectangle(image, coordinates, color=(0, 255, 0), thickness=2):
    """Draw a rectangle on the image."""
    import cv2
    x1, y1, x2, y2 = coordinates
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def put_text(image, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    """Put text on the image."""
    import cv2
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
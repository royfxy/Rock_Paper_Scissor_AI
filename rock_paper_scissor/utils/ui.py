import cv2

def text_animation(text, iter, image, total = 15, color = (255, 255, 0)):
    """
    Display text with animation on the center of the screen
    
    Parameters:
    text (str): text to display
    iter (int): current iteration
    image (numpy.ndarray): image to display text on
    total (int): total number of iterations
    color (tuple): color of text
    """
    font_size = 15/(iter%total+5) + 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 3)
    image_h = image.shape[0]
    image_w = image.shape[1]
    pos_h = int(image_h + text_size[0][1])/2
    pos_w = int(image_w - text_size[0][0])/2
    cv2.putText(image, text, (int(pos_w), int(pos_h)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 3)
    return image
from PIL import Image
import matplotlib.pyplot as plt


def check_yes_no(text):
    text = text.strip().lower()
    if text.startswith("yes"):
        return 1
    elif text.startswith("no"):
        return 0
    else:
        return None
    
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException
    
def disp_img(img_path): 
    img=Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')  
    plt.show()
    img.close()
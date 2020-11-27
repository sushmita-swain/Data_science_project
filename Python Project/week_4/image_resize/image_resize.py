# Pillow is a fork of the Python Imaging Library (PIL). PIL is a library that offers several standard procedures for manipulating images.
# It supports a range of image file formats such as PNG, JPEG, PPM, GIF, TIFF and BMP. 


# Installation
# pip3 install Pillow


# The Image Object

# A crucial class in the Python Imaging Library is the Image class. It is defined in the Image module and provides 
# a PIL image on which manipulation operations can be carried out

# To load an image from a file, we use the open() function in the Image module passing it the path to the image.



# from tkinter import Listbox
# from PIL import Image
# import tkinter as tk
# import glob

# def shrink(link, r=2):
#     img = Image.open(link)
#     w,h = img.size
#     w = w//r
#     h = h//r
#     img = img.resize((w,h),Image.ANTIALIAS)
#     return img



# class GUI:
#     def __init__(self,image):
#         self.image = image
#         root = tk.Tk()
#         root.title("Resizing image")
#         img = tk.PhotoImage(file = self.image)
#         label = tk.Label(root, image= img)
#         label.pack()
#         listbox = tk.Listbox(root)
#         listbox.pack(fill=tk.BOTH)
#         for file in glob.glob("*.png"):
#             listbox.insert(0,file)

#         lab2 = tk.Label(root, text="Insert the ratio to shrink image (2-50%)")
#         lab2.pack()    
#         entry = tk.Entry(root)
#         entry.pack() 
#         button = tk.Button(root, text = "Click here to shrink") 
#         button.pack()  
#         root.mainloop()


# app = GUI("/home/sushmita/Downloads/python-project/Python Project/week_4/images/image1.jpeg")        



from PIL import Image

image = Image.open('images/image1.jpeg')

image.show()

# The file format of the source file.
print(image.format) # Output: JPEG

# The pixel format used by the image. Typical values are “1”, “L”, “RGB”, or “CMYK.”
print(image.mode) # Output: RGB

# Image size, in pixels. The size is given as a 2-tuple (width, height).
print(image.size) # Output: (1200, 776)

# Colour palette table, if any.
print(image.palette) # Output: None

# Changing Image Type

image = Image.open('images/image1.jpeg')
image.save('images/new_image.png')

# Resizing Images

image = Image.open('images/image1.jpeg')
new_image = image.resize((200, 200))
new_image.save('image_400.jpg')

print(image.size) # Output: (1200, 776)
print(new_image.size) # Output: (400, 400)


image = Image.open('images/image1.jpeg')
image.thumbnail((400, 400))
image.save('image_thumbnail.jpg')

print(image.size) # Output: (400, 258)


# Cropping

image = Image.open('images/image1.jpeg')
box = (150, 200, 600, 600)
cropped_image = image.crop(box)
cropped_image.save('cropped_image.jpg')


# Pasting an Image onto Another Image
image = Image.open('images/image1.jpeg')
logo = Image.open('images/image2.jpeg')
image_copy = image.copy()
position = ((image_copy.width - logo.width), (image_copy.height - logo.height))
image_copy.paste(logo, position)
image_copy.save('pasted_image.jpg')
image_copy.paste(logo, position, logo)


# Rotating Images

image = Image.open('images/image1.jpeg')

image_rot_90 = image.rotate(90)
image_rot_90.save('image_rot_90.jpg')

image_rot_180 = image.rotate(180)
image_rot_180.save('image_rot_180.jpg')

image.rotate(18).save('image_rot_18.jpg')
image.rotate(18, expand=True).save('image_rot_18.jpg')

# Flipping Images

image = Image.open('images/image1.jpeg')

image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
image_flip.save('image_flip.jpg')

# Drawing on Images

from PIL import Image, ImageDraw

blank_image = Image.new('RGBA', (400, 300), 'white')
img_draw = ImageDraw.Draw(blank_image)
img_draw.rectangle((70, 50, 270, 200), outline='red', fill='blue')
img_draw.text((70, 250), 'Hello World', fill='green')
blank_image.save('drawn_image.jpg')

# Color Transforms

image = Image.open('images/image1.jpeg')

greyscale_image = image.convert('L')
greyscale_image.save('greyscale_image.jpg')
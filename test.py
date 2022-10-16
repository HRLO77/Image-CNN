from PIL import Image
# print(i%2)[0, 200, 0, 300] (==0), [100, 0, 200, 0] (==1), [100, 200, 200, 300] (2 comp), [200, 200, 200, 400] (1 comp)

image = Image.open('./images/dog_20.jpeg')

image = image.crop((100, 0, 200, 300))

s=''
for y in range(image.height):
    for x in range(image.width):
        color = image.getpixel((x, y))
        s += f"\033[48;2;{';'.join(str(i) for i in color)}m   " # images may be a bit stretched, change this accordingly
    s +='\033[0mㅤ\n'
    
print(s)
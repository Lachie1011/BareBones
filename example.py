# An example script that utilises the yoloV5 Lite

# Importing libraries
from parameters import Parameters
from barebones import BareBones

pt = Parameters("C:/Users/lachl/OneDrive/Desktop/YBV2.pt")

barebones = BareBones(pt)

image = "C:/Users/lachl/OneDrive/Desktop/test1.jpg"

results = barebones.inference(pt, image)





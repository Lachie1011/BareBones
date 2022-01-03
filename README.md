# BareBones 

**To be used with the YoloV5 main directory** <br/><br/>
A refactoring of the detect.py script so that it can be used like the PyTorch wrapper but allow access to all the parameters within the detect.py script in YoloV5. As well as removing redundant aspects such as the plotting functionalties so that higher dependencies like 'Matplotlib' arent required - easier to be used on ARM systems like the Jetson Nano. <br/><br/>

Utilises a new parameters classes that allows the same cmd line parameters in the detect.py script to be passed into the main inferencing functions. <br/><br/>

See 'example.py for usage'<br/>

By Lachlan Masson <br/>

**Dataset GWFONTS**

To access GWFONTS dataset, unzip converted.zip file.
This folder is divided in train (~80%) and test (~20%) parts.
Each of these directories contains several typefaces.
For each typeface there exists single folder which name is the same as original typeface name.
The folder contains 62 .png files for letters a-z, A-Z and digits 0-9.
The file for each character is named "*_folder-name_*_*_ascii-code_*.png".

To create this dataset from the original one, put it into folder "datasets/" and run
"models/SeparatingStyleAndContent/ttf_to_png.py".

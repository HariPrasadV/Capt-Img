# Capt-Img
An Image Captioning tool  
  
Requirements:  
	0) CUDA must be enabled for both training and captioning.  
	1) Python3  
	2) pytorch and torchvision  
	3) matplotlib  
	4) nltk  
	5) numpy  
	6) Pillow  
	7) argparse  
	8) mscoco API for accessing data from coco dataset (installation instructions in Usage)  
  
Usage: (Derived the from the pytorch tutorial on Image Captioning)  
	1. Clone the repositories  
		$ git clone https://github.com/pdollar/coco.git  
		$ cd coco/PythonAPI/  
		$ make  
		$ python setup.py build  
		$ python setup.py install  
		$ cd ../../  
		$ git clone https://github.com/HariPrasadV/Capt-Img.git  
		$ cd Capt-Img  
	2. Download the dataset  
		$ chmod +x download.sh  
		$ ./download.sh  
	3. Preprocessing (Instead of building the vocabulary from scratch you can download a pre-trained vocabulary file from the link given below)  
		$ python3 build_vocab.py  
		$ python3 resize.py  
	4. Train the model (All the files take arguments. Look into them for more info. Instead of training the models you can choose to download the pre-trained models from the links below)  
		$ python3 train-p.py  
		$ python3 train-e.py    
		$ python3 train-v.py    
	5. Test the model  
		$ python3 sample.py --image='png/example.png'  
  
Pre-Trained models :  
	Vocabulary 		- https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0  
	P-net models 	- https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0  
	E-net models 	- https://drive.google.com/open?id=1fshQm4uMm19J_jUMXmZE3wgMK7wCN0Ab  
	V-net models 	- https://drive.google.com/open?id=1aoVq0nnaOcL5u_3vUixZdz7Y5KEcfN9a

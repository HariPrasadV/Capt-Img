mkdir data
wget http://images.cocodataset.org/zips/train2017.zip -P ./data/
wget http://images.cocodataset.org/zips/val2017.zip -P ./data/
wget http://images.cocodataset.org/zips/test2017.zip -P ./data/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./data/

unzip ./data/train2017.zip -d ./data/
rm ./data/train2017.zip 
unzip ./data/val2017.zip -d ./data/ 
rm ./data/val2017.zip
unzip ./data/test2017.zip -d ./data/ 
rm ./data/test2017.zip 
unzip ./data/annotations_trainval2017.zip -d ./data/
rm ./data/annotations_trainval2017.zip
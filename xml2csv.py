import os
import glob
import pandas as pd 
import xml.etree.ElementTree as ET 

def xml2csv(path_to_xml):
	list_of_xml_files = []
	for file in glob.glob(path_to_xml + '/*.xml'):
		tree = ET.parse(file)
		root = tree.getroot()
		for member in root.findall('object'):
			val =  (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
			list_of_xml_files.append(val)
	col = ['filename','width','height','class','xmin','ymin','xmax','ymax']
	frame_xml = pd.DataFrame(list_of_xml_files,columns=col)
	return frame_xml

def main():
	for folders in ['train','test']:
	    img_path = os.path.join(os.getcwd(),('images/'+folders))
	    frame_xml = xml2csv(img_path)
	    frame_xml.to_csv(('images/'+folders+'_labels.csv'),index=None)
	print('Done')

main()



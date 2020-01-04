import numpy as np
import os

'''
def main():
	
	dirpath = os.path.join('/mrtstorage/users/chli/data_road/training/gt_image_2')
	os.chdir('/mrtstorage/users/chli/data_road/training/gt_image_2')
	print(os.listdir(dirpath))
	for filename in os.listdir(dirpath):
		print(filename)

		name = filename.split('.')[0]
		if name.split('_')[1]=='road':
			print(filename.split('_'))
			id1 = name.split('_')[0]
			id2 = name.split('_')[2]
			id = id1 + '_' +id2+ '.png'
			os.rename(filename, id )
'''
def main():
	dirpath = os.path.join('/mrtstorage/users/chli/data_road/testing/image_2')
	os.chdir('/mrtstorage/users/chli/data_road/testing/image_2')
	print(os.listdir(dirpath))
	i= 0
	for filename in os.listdir(dirpath):

		os.rename(filename, str(i)+'.png' )
		i=i+1



		

if __name__ == '__main__':
    main()

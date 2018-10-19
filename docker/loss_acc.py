import sys
import numpy as np
import visdom
import os

default_path = './result/'

def plot_loss_acc(record_path = default_path):
	'''
	用于训练过程的可视化，包括train、test的loss和accuracy的变化趋势；
	运行之前命令行执行：python -m visdom.server
	根据提示端口在浏览器查看可视化结果

	:param record_path: 保存的训练log所在的路径
	:return: None
	'''

	files = os.listdir(record_path)
	viz = visdom.Visdom()

	for file in files:
		file_name_list = file.split('_')
		if(os.path.splitext(file)[1] == '.txt' and len(file_name_list) > 2 and file_name_list[0] == 'train'):
			train_file = os.path.join(record_path, 'train_' + '_'.join(file_name_list[1:]))
			test_file = os.path.join(record_path, 'test_' + '_'.join(file_name_list[1:]))
			train_record = np.loadtxt(train_file)
			test_record = np.loadtxt(test_file)
			epoch_len = len(train_record)

			if('loss' in file_name_list[2]):
				viz.env = 'loss'
				viz.line(X=np.column_stack((np.array(range(epoch_len)), np.array(range(epoch_len)))),
						 Y=np.column_stack((np.array(train_record), np.array(test_record))),
						 opts=dict(
							 legend=["train_loss", "test_loss"],
							 xlabel='epoch',
							 title='_'.join(file_name_list[1:3])
						 ))
			elif('acc' in file_name_list[2]):
				viz.env = 'acc'
				viz.line(X=np.column_stack((np.array(range(epoch_len)), np.array(range(epoch_len)))),
						 Y=np.column_stack((np.array(train_record), np.array(test_record))),
						 opts=dict(
							 legend=["train_acc", "test_acc"],
							 xlabel='epoch',
							 title='_'.join(file_name_list[1:3])
						 ))

			print(file + ' has been ploted.') 

if __name__ == '__main__':
		plot_loss_acc()

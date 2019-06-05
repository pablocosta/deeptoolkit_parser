from reader import *
import argparse
import os
import time
from torchtext import data


def main(args):

	# Load dataset
	train_file = os.path.join(args.data_path, "data_train_{}.json".format(args.max_len))
	dev_file   = os.path.join(args.data_path, "data_dev_{}.json".format(args.max_len))
	test_file  = os.path.join(args.data_path, "data_test_{}.json".format(args.max_len))

	start_time = time.time()
	if os.path.isfile(train_file) and os.path.isfile(dev_file) and os.path.isfile(test_file):
		print ("Loading data..")
		dp_train = DataPreprocessor()
		dp_dev   = DataPreprocessor()
		dp_test  = DataPreprocessor()

		train_dataset, vocabs_train = dp_train.load_data(train_file)
		dev_dataset, vocabs_dev     = dp_dev.load_data(dev_file)
		test_dataset, vocabs_test   = dp_test.load_data(test_file)

	else:
		print ("Preprocessing data..")
		dp_train = DataPreprocessor()
		dp_dev   = DataPreprocessor()
		dp_test  = DataPreprocessor()

		train_dataset, vocabs_train = dp_train.preprocess(os.path.join(args.dataset_path, args.train_files), train_file, args.max_len)
		dev_dataset, vocabs_dev     = dp_dev.preprocess(os.path.join(args.dataset_path, args.dev_files), dev_file, args.max_len)
		test_dataset, vocabs_test   = dp_test.preprocess(os.path.join(args.dataset_path, args.test_files), test_file, args.max_len)

	print ("Elapsed Time: %1.3f \n" % (time.time() - start_time))

	print("=========== Data Stat ===========")
	print("Train: ", len(train_dataset))
	print("Dev: ", len(dev_dataset))
	print("Test: ", len(test_dataset))
	print("=================================")



	train_loader = data.BucketIterator(dataset=train_dataset, batch_size=args.batch_size,
									   repeat=False, shuffle=True, sort_within_batch=True,
									   sort_key=lambda x: len(x.src))
	dev_loader = data.BucketIterator(dataset=dev_dataset, batch_size=args.batch_size,
									 repeat=False, shuffle=True, sort_within_batch=True,
									 sort_key=lambda x: len(x.src))
	test_loader = data.BucketIterator(dataset=test_dataset, batch_size=args.batch_size,
									 repeat=False, shuffle=True, sort_within_batch=True,
									 sort_key=lambda x: len(x.src))
	"""
		trainer = Trainer(train_loader, val_loader, vocabs, args)
		trainer.train_iters()

	"""

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Language setting

	parser.add_argument('--train_files', type=str, default='train.txt')
	parser.add_argument('--dev_files', type=str, default='dev.txt')
	parser.add_argument('--test_files', type=str, default='test.txt')
	parser.add_argument('--max_len', type=int, default=100)

	# Model hyper-parameters
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--grad_clip', type=float, default=2)
	parser.add_argument('--num_layer', type=int, default=2)

	#embedding dimension and hideen dimension must be equals
	parser.add_argument('--embed_dim', type=int, default=600)
	parser.add_argument('--hidden_dim', type=int, default=600)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--attn_model', type=str, default="general")
	parser.add_argument('--decoder_lratio', type=float, default=5.0)
	parser.add_argument('--early_stoping', type=int, default=20)
	parser.add_argument('--evaluate_every', type=int, default=100)


	# Training setting
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_epoch', type=int, default=2650)
	parser.add_argument('--teacher_forcing', type=float, default=1.0)

	# Path
	parser.add_argument('--data_path', type=str, default='./data_path/')
	parser.add_argument('--dataset_path', type=str, default='./data/')

	# Dir.
	parser.add_argument('--log', type=str, default='log')
	parser.add_argument('--sample', type=str, default='sample')

	# Misc.
	parser.add_argument('--gpu_num', type=int, default=0)

	args = parser.parse_args()
	print (args)
	main(args)
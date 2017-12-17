
mkdir 'data_notMNIST'

wget 'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz'
wget 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz'

git clone 'git@github.com:davidflanagan/notMNIST-to-MNIST.git'

python convert_to_mnist_format.py notMNIST_small 1800 t10k-labels-idx1-uyte t10k-images-idx3-ubyte
python convert_to_mnist_format.py notMNIST_large 50000 data/train-labels-idx1-byte data/train-images-idx3-ubyte

gunzip -d -k *.gz

import preprocess
import postprocess



if __name__ == "__main__":
    path = 'data'

    images, labels = preprocess.load_mnist(path=path)
    postprocess.plot_multiple_digit(images[0:200,:,:])

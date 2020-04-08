from utils import parse_args, create_experiment_dirs, calculate_flops
from model import MobileNet
from train import Train
from data_loader import DataLoader
from summarizer import Summarizer
import tensorflow as tf
from crop_face import FaceCropper
import cv2

def main():
    # Parse the JSON arguments
    try:
        config_args = parse_args()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)

    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Data loading
    data = DataLoader(config_args.batch_size, config_args.shuffle)
    print("Loading Data...")
    config_args.img_height, config_args.img_width, config_args.num_channels, \
    config_args.train_data_size, config_args.test_data_size = data.load_data()
    print("Data loaded\n\n")

    # Model creation
    print("Building the model...")
    model = MobileNet(config_args)
    print("Model is built successfully\n\n")

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)
    # Train class
    trainer = Train(sess, model, data, summarizer)

    # if config_args.to_train:
    #     try:
    #         print("Training...")
    #         trainer.train()
    #         print("Training Finished\n\n")
    #     except KeyboardInterrupt:
    #         trainer.save_model()

    # if config_args.to_test:
    #     print("Final test!")
    #     trainer.test('val')
    #     print("Testing Finished\n\n")
    # trainer.dectect(FaceCropper().generate('fake.png'))

if __name__ == '__main__':
    # main()
    config_args = parse_args()
    config_args.img_height, config_args.img_width, config_args.num_channels = (224, 224, 3)
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    faces = FaceCropper().generate('maxresdefault.jpg')
    with tf.Session(config=config) as sess:
        config_args.batch_size = len(faces)
        model = MobileNet(config_args)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver(max_to_keep=config_args.max_to_keep,
                                    keep_checkpoint_every_n_hours=10,
                                    save_relative_paths=True)
        saver.restore(sess, tf.train.latest_checkpoint(config_args.checkpoint_dir))

        # show camera
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            croppedFaces = []
            for (x, y, w, h) in faces:
                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)

                faceimg = img[ny:ny+nr, nx:nx+nr]
                lastimg = cv2.resize(faceimg, (224, 224))
                croppedFaces.append(lastimg)
        
            normalizedFaces = np.array(croppedFaces).reshape((len(croppedFaces), 224, 224, 3))
            results = sess.run(model.y_out_argmax, feed_dict={model.X: normalizedFaces, model.is_training: False})
            i = 0
            for (x, y, w, h) in faces:
                if (results[i] == 0):
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                    cv2.putText(img, 'Fake', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
                if (results[i] == 1):
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(img, 'Fake', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
        cap.release()
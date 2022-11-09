import argparse
from rock_paper_scissor.app.game import Game
from rock_paper_scissor.network.gesture_prediction.data_capture import capture_data
from rock_paper_scissor.network.gesture_prediction.train import train_model, k_fold_cross_validation_train
from rock_paper_scissor.network.gesture_recognition.Train_from_csv import train_model_static

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='game', help='run game or train model')
    parser.add_argument('--model_name', type=str, default='model.pth', help='model path')
    parser.add_argument('--gesture', type=str, default='paper', help='gesture to capture')
    parser.add_argument('--k_fold', type=int, default=1, help='k fold cross validation')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    mode = args.run
    prediction_model_pth = "rock_paper_scissor/network/gesture_prediction/checkpoints/" + args.model_name + ".pth"
    recognition_model_pth = "rock_paper_scissor/network/gesture_recognition/checkpoints/" + args.model_name + ".pth"
    if mode == 'game':
        game = Game(prediction_model_pth, recognition_model_pth)
        game.play()
    elif mode == 'data_cap':
        file_pth = "rock_paper_scissor/network/gesture_prediction/data/" + args.gesture + ".npy"
        capture_data(file_pth, 3)
    elif mode == 'train':
        if args.k_fold == 1:
            train_model(prediction_model_pth, args.epochs)
        else:
            k_fold_cross_validation_train(args.k_fold, prediction_model_pth, args.epochs)
    elif mode == 'train_recognition':
        train_model_static(recognition_model_pth, args.epochs)
    else:
        print("invalid mode")

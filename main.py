import argparse
from rock_paper_scissor.app.game import Game
from rock_paper_scissor.network.gesture_prediction.data_capture import capture_data
from rock_paper_scissor.network.gesture_prediction.train import train_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='game', help='run game or train model')
    parser.add_argument('--model_name', type=str, default='model.pth', help='model path')
    parser.add_argument('--gesture', type=str, default='paper', help='gesture to capture')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    mode = args.run
    prediction_model_pth = "rock_paper_scissor/network/gesture_prediction/checkpoints/" + args.model_name
    recognition_model_pth = "rock_paper_scissor/network/gesture_recognition/checkpoints/" + args.model_name
    if mode == 'game':
        game = Game(prediction_model_pth, recognition_model_pth)
        game.play()
    elif mode == 'data_cap':
        file_pth = "rock_paper_scissor/network/gesture_prediction/data/" + args.gesture + ".npy"
        capture_data(file_pth, 3)
    elif mode == 'train':
        train_model(prediction_model_pth)
    else:
        print("invalid mode")
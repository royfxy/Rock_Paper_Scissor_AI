import argparse
from rock_paper_scissor.app.game import Game

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='game', help='run game or train model')
    parser.add_argument('--model_name', type=str, default='model.pth', help='model path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    mode = args.run
    prediction_model_pth = "rock_paper_scissor/network/gesture_prediction/checkpoints/" + args.model_name
    recognition_model_pth = "rock_paper_scissor/network/gesture_recognition/checkpoints/" + args.model_name
    if mode == 'game':
        game = Game(prediction_model_pth, recognition_model_pth)
        game.play()
    elif mode == 'train':
        print("train")
    else:
        print("invalid mode")
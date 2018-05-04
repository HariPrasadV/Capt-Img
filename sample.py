import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, EncoderRNN, DecoderRNN, MLPNN, path_to_data
from PIL import Image


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build Models
    encoder_polc = EncoderCNN(args.embed_size)
    encoder_polc.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder_polc = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    encoder_val_img = EncoderCNN(args.hidden_size)
    encoder_val_img.eval()  # evaluation mode (BN uses moving mean/variance)
    encoder_val_capt = EncoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)
    mlp_val = MLPNN(args.hidden_size+args.hidden_size)

    # Load the trained model parameters
    encoder_polc.load_state_dict(torch.load(args.encoder_path_p))
    decoder_polc.load_state_dict(torch.load(args.decoder_path_p))

    encoder_val_img.load_state_dict(torch.load(args.encoder_path_v_img))
    encoder_val_capt.load_state_dict(torch.load(args.encoder_path_v_capt))
    mlp_val.load_state_dict(torch.load(args.mlp_path_v))

    # Prepare Image
    image = load_image(args.image, transform)
    image_tensor = to_var(image, volatile=True)

    # If use gpu
    if torch.cuda.is_available():
        encoder_polc.cuda()
        decoder_polc.cuda()
        encoder_val_img.cuda()
        encoder_val_capt.cuda()
        mlp_val.cuda()

    # Generate caption from image
    feature = encoder_polc(image_tensor)
    batch_size = 2
    sampled_ids = decoder_polc.sample(feature, image_tensor, encoder_val_img, encoder_val_capt, mlp_val, batch_size, args.lambda_l)
    sampled_ids = sampled_ids.cpu().data.numpy()

    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out image and generated caption.
    print (sentence)
    image = Image.open(args.image)
    #print(np.asarray(image))
    plt.imshow(np.asarray(image))
    title_obj = plt.title(sentence) #get the title property handler
#    plt.getp(title_obj)                    #print out the properties of title
#    plt.getp(title_obj, 'text')            #print out the 'text' property for title
    plt.setp(title_obj, color='r')         #set the color of title to red
    plt.pause(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')

    parser.add_argument('--encoder_path_p', type=str, default='./models/policyNet/encoder-5-3000.pkl',
                        help='path for trained policy encoder')
    parser.add_argument('--decoder_path_p', type=str, default='./models/policyNet/decoder-5-3000.pkl',
                        help='path for trained policy decoder')

    #parser.add_argument('--encoder_path_e_img', type=str, default='./models/rewardNet/encoder-img-5-3000.pkl',
    #                    help='path for trained reward encoder for images')
    #parser.add_argument('--encoder_path_e_capt', type=str, default='./models/rewardNet/encoder-capt-5-3000.pkl',
    #                    help='path for trained reward encoder for captions')

    parser.add_argument('--encoder_path_v_img', type=str, default='./models/valueNet/encoder-img-1-79000-v.pkl',
                        help='path for trained value encoder for images')
    parser.add_argument('--encoder_path_v_capt', type=str, default='./models/valueNet/encoder-capt-1-79000-v.pkl',
                        help='path for trained value encoder for captions')
    parser.add_argument('--mlp_path_v', type=str, default='./models/valueNet/mlp-1-79000-v.pkl',
                        help='path for trained mlp')

    parser.add_argument('--vocab_path', type=str, default=path_to_data+'vocab.pkl',
                        help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--lambda_l', type=float , default=1.0 ,
                        help='weight given to value prediction')
    args = parser.parse_args()
    main(args)

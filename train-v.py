import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, EncoderRNN, path_to_data, MLPNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main(args):
    random.seed()
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build the models
    encoder_img = EncoderCNN(args.hidden_size)
    encoder_capt = EncoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)
    mlp = MLPNN(args.hidden_size+args.hidden_size)

    encoder_img_e = EncoderCNN(args.hidden_size)
    encoder_capt_e = EncoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    # load the reward model
    encoder_img_e.load_state_dict(torch.load(args.encoder_path_e_img))
    encoder_capt_e.load_state_dict(torch.load(args.encoder_path_e_capt))

    if torch.cuda.is_available():
        encoder_img.cuda()
        encoder_capt.cuda()
        mlp.cuda()
        encoder_img_e.cuda()
        encoder_capt_e.cuda()

    # Loss and Optimizer
    criterion = nn.MSELoss()
    params = list(encoder_capt.parameters()) + list(encoder_img.linear.parameters()) + list(encoder_img.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)

            features = encoder_img_e(images)
            outputs = encoder_capt_e(captions, lengths)
            scores = torch.mm(features, outputs.transpose(1, 0))
            diagonal = scores.diag()

            #rvals = torch.ones(images.size[0]) # batchlength size
            rvals = diagonal.detach() # batchlength size
            #rvals = torch.autograd.Variable(diagonal, requires_grad=False)
            # targets = pack_padded_sequence(rvals, lengths, batch_first=True)[0]
            # Forward, Backward and Optimize
            encoder_capt.zero_grad()
            encoder_img.zero_grad()
            mlp.zero_grad()

            img_features = encoder_img(images)
            #TODO randomly convert the caption to be partial
            n = captions[0].size(0)
            t = n*torch.rand(captions.size(0), device=torch.device("cuda"))
            t = t.type(torch.long)
            for k in range(captions.size(0)):
                #print("t[",k,"]=",t[k])
                if t[k] < lengths[k]:
                    captions[k][t[k]] = 2
                captions[k][t[k]+1:n] = torch.zeros(n-int(t[k])-1, device=torch.device("cuda"))
            lengths = t+1

            lengths, indices = torch.sort(torch.tensor(lengths),descending=True)
            captions.index_copy_(0,indices,captions)
            img_features.index_copy_(0,indices,img_features)
            rvals.index_copy_(0,indices,rvals)

            cap_features = encoder_capt(captions, lengths)
            outputs = mlp(img_features, cap_features)

            loss = criterion(outputs, rvals)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step,
                        loss.data[0], np.exp(loss.data[0])))

            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(encoder_capt.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-capt-%d-%d-v.pkl' %(epoch+1, i+1)))
                torch.save(encoder_img.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-img-%d-%d-v.pkl' %(epoch+1, i+1)))
                torch.save(mlp.state_dict(),
                           os.path.join(args.model_path,
                                        'mlp-%d-%d-v.pkl' %(epoch+1, i+1)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/valueNet/' ,
                        help='path for saving trained models')

    parser.add_argument('--encoder_path_e_img', type=str, default='./models/rewardNet/encoder-img-1-37000.pkl',
                        help='path for trained reward encoder for images')
    parser.add_argument('--encoder_path_e_capt', type=str, default='./models/rewardNet/encoder-capt-1-37000.pkl',
                        help='path for trained reward encoder for captions')

    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default=path_to_data+'vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default=path_to_data+'resized2017' ,
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default=path_to_data+'annotations/captions_train2017.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)

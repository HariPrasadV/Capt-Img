import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

path_to_data = '/home/hp/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/'

class EncoderCNN(nn.Module):
    """ Pre-trained CNN(Resnet-152) which is used CNN_p, CNN_e and CNN_v"""
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class MLPNN(nn.Module):
    """ MLP used to predict the reward """
    def __init__(self, input_size):
        super(MLPNN, self).__init__()
        input_size = int(input_size)
        self.fc1 = nn.Linear(input_size, int(input_size/2))
        self.fc2 = nn.Linear(int(input_size/2), 1)

    def forward(self, img_features, cap_features):
        x = torch.cat((img_features, cap_features), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0))
        return x

class EncoderRNN(nn.Module):
    """ RNN embedding the input setence which is used as RNN_v and RNN_e"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(EncoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """ generates hT(S') """
        embeddings = self.embed(captions)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        outputs, (hidden, cell) = self.lstm(packed)
        return hidden[0]

class DecoderRNN(nn.Module):
    """ RNN for generating sentences, used as RNN_p"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, image_tensor, model_encoder_img, model_encoder_capt, model_mlp, batch_size, L, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        K=3
        best_caption = torch.tensor([1])
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size),
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            sampled_ids_topk = []
            #print("outputs=",outputs)
            #print("outputs=",outputs.max(1))
            #print("outputs=",outputs.topk(k))
            max_k, predicted_k = outputs.topk(K)
            predicted_k = predicted_k[0]
            max_k = max_k[0]
            #print("max_k=",max_k)
            #print("predicted_k=",predicted_k)
            #print("predicted_k=",predicted_k)
            predicted = outputs.max(1)[1]
            #print("predicted=",predicted)
            if i>0:
                features_v = model_encoder_img(image_tensor)
                captions = []
                lengths = [i+1]*batch_size
                max_rewards_index = -1
                max_reward = -float('inf')
                for kk in range(K):
                    captions.append(best_caption.clone())
                    #print("captions[kk]=",captions[kk])
                    captions[kk] = captions[kk].resize_(i+1)
                    #print("captions[kk]=",captions[kk])
                    captions[kk][i] = predicted_k[kk]
                    #print("captions[kk]=",captions[kk])
                    #captions[kk][i+1] = 2
                    #print("captions[kk]=",captions[kk])
                    captions[kk] = captions[kk].cuda()
                    capt_batch = captions[kk].expand(batch_size, i+1)
                    outputs_v = model_encoder_capt(capt_batch, lengths)
                    scores_v = torch.mm(features_v, outputs_v.transpose(1, 0))
                    diagonal_v = scores_v.diag()
                    #print("diagonal_v=",diagonal_v)
                    curr_score = L*diagonal_v[0] + (1.0-L)*max_k[kk]
                    if kk==0 or curr_score > max_reward:
                        max_rewards_index = kk
                        max_reward = curr_score

                #print("max_rewards_index=",max_rewards_index)
                #print("max_reward=",max_reward)
                predicted[0] = predicted_k[max_rewards_index]
                best_caption = best_caption.resize_(i+1)
                best_caption[i] = predicted
                #print("best_caption=",best_caption)
            #TODO try top k possibilities
            #print("predicted=",predicted)
            #predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()

class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):
        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        diagonal = scores.diag()
        #print("diagonal scores:",diagonal)
        #print("diagonalS:",diagonal.size())

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return cost_s.sum() + cost_im.sum()


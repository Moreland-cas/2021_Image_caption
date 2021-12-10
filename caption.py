import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from PIL import Image

device = torch.device("cpu")


def caption_a_image(encoder, decoder, image_path, word_map, beam_size=3):
    """
    :return: captions, weights for visualization
    """
    k = beam_size
    vocab_size = len(word_map)

    # Read image and preprocess it
    image = Image.open(image_path).convert("RGB")
    T = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = T(image)

    image = image.unsqueeze(0)
    # 1 * 3 * 224 * 224
    assert image.size(1) == 3
    encoder_out = encoder(image)
    # 1 * enc_image_size * enc_image_size * encoder_dim
    enc_image_size = encoder_out.size(1)
    assert enc_image_size == 14
    encoder_dim = encoder_out.size(3)

    encoder_out = encoder_out.view(1, -1, encoder_dim)
    # 1 * num_pixels * encoder_dim
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    # treat it as a batch k feed forwarding

    k_prev_words = torch.LongTensor([[word_map["<start>"]]] * k).to(device)
    # store previous word, k * 1, list as its element
    seqs = k_prev_words
    # store previous sequences, k * 1
    top_k_scores = torch.zeros(k, 1).to(device)
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s <= k, once a sequence with <end> is selected, it is removed
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        # s, embed_dim
        awe, alpha = decoder.attention(encoder_out, h)
        # decoder.attention() returns weighted feature and attention map
        # in shape b * encoder_dim, b * num_pixels respectively
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        # s * enc_image_size * enc_image_size
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        # s * decoder_dim

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)
        # s * vocab_size
        # after log operation, probability of product turns into sum

        scores = top_k_scores.expand_as(scores) + scores
        # top_k_scores in shape s * 1
        # scores in shape s * v, s rows stand for s previous state, v for vocab_size choices
        # choose the top k out of s * v choice

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            # s, s, [value1, value2, ..., values], [index1, index2, ..., indexs]
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size
        # s, s  prev means which sentence, next means which word

        # Add new words to sequences, alphas
        # seqs in shape s * step
        # seqs_alpha in shape s * step * enc_img_size * enc_img_size
        # print(top_k_words, prev_word_inds)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        # filter the completed sequence
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map["<end>"]]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            # num_newly_complete * length
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            print(seqs)
            break
        step += 1
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    return seq, alphas


if __name__ == "__main__":
    model_path = "./CKPT/ckpt_epoch_23_bleu4_0.22502506266757916.pth.tar"
    checkpoint = torch.load(model_path, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    word_map_path = "./word_map.json"
    with open(word_map_path, 'r') as f:
        word_map = json.load(f)

    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    for i in range(5):
        seq, alphas = caption_a_image(encoder,
                                    decoder,
                                    image_path="./test_image/%d.jpg" % (i + 1),
                                    word_map=word_map,
                                    beam_size=3)
        word_seq = [rev_word_map[item] for item in seq]
        print(seq)
        print(word_seq)











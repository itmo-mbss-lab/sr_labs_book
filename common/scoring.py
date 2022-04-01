# The script allows to extract embeddings, to compute scores between enroll and test speaker models for performing of test procedure


# Import of modules
import numpy
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import tqdm


def extract_features(model, test_loader):
    # Extract features for every waveform

    feats = {}

    for idx, data in enumerate(test_loader):
        inp1 = data[0][0].cuda()
        
        with torch.no_grad():
            ref_feat = model(inp1).detach().cpu()
        
        feats[data[1][0]] = ref_feat

    return feats


def compute_scores(feats, lines):
    """
    Compute scores by pairwise_distance for protocol lines
    """

    all_scores = []
    all_labels = []
    all_trials = []

    for idx, line in enumerate(lines):

        data = line.split()

        ref_feat = feats[data[1]].cuda()
        com_feat = feats[data[2]].cuda()

        ref_feat = F.normalize(ref_feat, p=2, dim=1)
        com_feat = F.normalize(com_feat, p=2, dim=1)

        dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()

        score = -1*numpy.mean(dist)
        
        all_scores.append(score)
        all_labels.append(int(data[0]))
        all_trials.append(data[1]+" "+data[2])

    return all_scores, all_labels, all_trials


def compute_scores_cosine(data, lines):
    """
    Compute scores by cosine metric for protocol lines
    """
    all_scores = []
    all_labels = []
    all_trials = []

    for idx, line in tqdm.tqdm(enumerate(lines), total=len(lines), desc='Scoring progress'):
        trial_label, enroll_wav, test_wav = line.split()
        E = data[enroll_wav].squeeze(0).numpy()
        T = data[test_wav].squeeze(0).numpy()

        E = E.reshape(1, -1)
        T = T.reshape(1, -1)
        score = cosine_similarity(E, T)

        all_scores.append(score[0][0])
        all_labels.append(int(trial_label))
        all_trials.append(enroll_wav + " " + test_wav)

    return all_scores, all_labels, all_trials

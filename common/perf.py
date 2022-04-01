# The script computes EER, minDCF and actDCF


# Import of modules
import numpy as np


def ecdf(x):
    # Additional functions for computation of verification metrics

    xs = np.sort(x, 0)
    ys = np.arange(1, len(xs) + 1 )/float(len(xs))

    return ys, xs

def get_eer(target_scores, imposter_scores):
    # Compute EER

    if target_scores.ndim == 2:
        
        if target_scores.shape[1] != 1:
            target_scores = target_scores.T
            imposter_scores = imposter_scores.T
        
        target_scores = np.squeeze(target_scores, 1)
        imposter_scores = np.squeeze(imposter_scores, 1)

    [ft, xt] = ecdf(target_scores)
    [fi, xi] = ecdf(imposter_scores)
    ft = ft[1:-1]
    xt = xt[1:-1]
    fi = fi[1:-1]
    xi = xi[1:-1]
    fi = 1 - fi

    x = np.sort(np.unique(np.append(xt, xi)))

    yt = np.interp(x, np.squeeze(xt), ft)
    yi = np.interp(x, np.squeeze(xi), fi)
    i = np.argmin(np.abs(yt - yi))

    EER = 100*yt[i]
    thresh_EER = x[i]
    
    if np.isnan(EER):
        EER = 0

    return EER, thresh_EER

def get_dcf(target_scores, imposter_scores, P_target = 1e-3):
    # Compute minDCF and actDCF

    if target_scores.ndim == 2:
        
        if target_scores.shape[1] != 1:
            target_scores = target_scores.T
            imposter_scores = imposter_scores.T
        
        target_scores = np.squeeze(target_scores, 1)
        imposter_scores = np.squeeze(imposter_scores, 1)

    [ft, xt] = ecdf(target_scores)
    [fi, xi] = ecdf(imposter_scores)
    ft = ft[1:-1]
    xt = xt[1:-1]
    fi = fi[1:-1]
    xi = xi[1:-1]
    fi = 1 - fi

    x = np.sort(np.unique(np.append(xt, xi)))

    yt = np.interp(x, np.squeeze(xt), ft)
    yi = np.interp(x, np.squeeze(xi), fi)

    C_miss = 1; C_fa = 1
    
    minDCF        = np.min(C_miss*yt*P_target + C_fa*yi*(1 - P_target))
    i             = np.argmin(C_miss*yt*P_target + C_fa*yi*(1 - P_target))
    C_def         = np.min([C_miss*P_target, C_fa*(1 - P_target)])
    minDCF        = minDCF/C_def
    thresh_minDCF = x[i]
    
    thresh_actDCF = np.log((1 - P_target)/P_target)
    i             = x.searchsorted(thresh_actDCF)
    
    if i == len(x):
        i = len(x) - 1
    
    actDCF        = C_miss*yt[i]*P_target + C_fa*yi[i]*(1 - P_target)
    actDCF        = actDCF/C_def

    return minDCF, thresh_minDCF, actDCF, thresh_actDCF
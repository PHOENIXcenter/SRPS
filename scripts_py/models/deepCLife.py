import tensorflow as tf
import numpy as np
import time, itertools
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

class deepCLife(Model):
    def __init__(self, layer_params):
        super(deepCLife, self).__init__()
        self.encoder_layers = [layers.Dense(
                layer_params['encoder_hdim'], 
                use_bias=layer_params['use_bias'], 
                activation=layer_params['activation'], 
                kernel_regularizer=regularizers.l1(l=layer_params['regularization'])
            ) 
            for _ in range(layer_params['encoder_layer_num'])
        ]
        self.class_layers = [layers.Dense(
                layer_params['class_hdim'], 
                use_bias=layer_params['use_bias'], 
                activation=layer_params['activation'], 
                kernel_regularizer=regularizers.l1(l=layer_params['regularization'])
            ) 
            for _ in range(layer_params['class_layer_num'] - 1)
        ]
        self.class_layers.append(layers.Dense(
                layer_params['subtype_num'], 
                use_bias=layer_params['use_bias'], 
                activation=None, 
                kernel_regularizer=regularizers.l1(l=layer_params['regularization'])
            )
        )
    
    def call(self, x, drop_rate=0., is_training=False):
        for layer in self.encoder_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        for layer in self.class_layers:
            x = layer(tf.nn.dropout(x, rate=drop_rate))
        return x

    def predict(self, x):
        x = self.call(x)
        a = tf.math.argmax(x, axis=1)
        return tf.squeeze(a).numpy()

    def predict_proba(self, x):
        x = self.call(x)
        p = tf.nn.softmax(x, axis=1)
        return tf.squeeze(p).numpy()

def make_models(layer_params, learning_rate):
    classifier = deepCLife(layer_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    return classifier, optimizer

# @tf.function
# def train_step(x1_list, y1_list, x2_list, OS_list, status_list, DFS_list, recurrence_list,
#     classifier, optimizer, loss_rates, dropout
# ):
#     with tf.GradientTape(persistent=True) as tape:
#         loss_su = 0
#         for x1, y1 in zip(x1_list, y1_list):
#             logits1 = classifier(x1, dropout)
#             loss_su = tf.reduce_mean(supervise_loss(logits1, y1))
#             loss_su += tf.reduce_mean(loss_su)

#         loss_survival = 0.
#         for x2, OS, status, DFS, recurrence in zip(x2_list, OS_list, status_list, DFS_list, recurrence_list):
#             logits2 = classifier(x2, dropout)
#             loss_OS = survival_divergence_loss(logits2, tf.cast(OS*10, tf.int32), status)
#             loss_DFS = survival_divergence_loss(logits2, tf.cast(DFS*10, tf.int32), recurrence)
#             loss_survival += (loss_OS + loss_DFS)/2

#         loss_l1 = tf.add_n(classifier.losses)

#         loss = tf.reduce_mean(loss_su) * loss_rates[0] + loss_survival * loss_rates[1] + loss_l1

#     gradients = tape.gradient(loss, classifier.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))
#     return loss_su, loss_survival, loss_l1

@tf.function
def train_step(x1, y1, x2, OS, status, DFS, recurrence, classifier, optimizer, loss_rates, dropout, lr):

    with tf.GradientTape(persistent=True) as tape:
        # supervised loss
        logits = classifier(x1, dropout)
        loss_su = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits))

        # survival loss
        prob = tf.nn.softmax(classifier(x2, dropout))
        loss_surv_OS = survival_divergence_loss(prob, tf.cast(OS * 10, dtype=tf.int32), status)
        loss_surv_DFS = survival_divergence_loss(prob, tf.cast(DFS * 10, dtype=tf.int32), recurrence)
        loss_surv = (loss_surv_OS + loss_surv_DFS) / 2

        # l1 normalization loss
        loss_l1 = tf.add_n(classifier.losses)

        loss = loss_su * loss_rates[0] + loss_surv * loss_rates[1] + loss_l1

    gradients = tape.gradient(loss, classifier.trainable_variables)
    optimizer.learning_rate = lr
    optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))

    return loss_su, loss_surv, loss_l1

@tf.function
def train_step_clustering(x, times, events, classifier, optimizers, dropout):
    with tf.GradientTape(persistent=True) as tape:
        logits = classifier(x, dropout)
        loss = survival_divergence_loss(logits, tf.cast(times*100, tf.int32), events)
    gradients = tape.gradient(loss, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))
    return loss

def supervise_loss(logits, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

def survival_divergence_loss(logits, lifetime, dead, nPairs=None):
    outs = tf.nn.softmax(logits)
    distros = {}
    k = outs.shape[1]

    nSamples = tf.reduce_sum(outs, axis=0)
    allPairs = list(itertools.combinations(range(k), 2))

    if nPairs is None or nPairs == "all":
        nPairs = len(allPairs)
    elif nPairs == "k":
        nPairs = k
    else:
        nPairs = int(nPairs)

    nClusterPairsToSample = min(nPairs, len(allPairs))

    clusterPairsToSample = np.random.choice(
        len(allPairs), nClusterPairsToSample, replace=False
    )
    clusterPairsToSample = [allPairs[i] for i in clusterPairsToSample]

    uniqueClusters = np.unique(
        [pair[i] for pair in allPairs for i in range(2)]
    )
    for ci in uniqueClusters:
        distros[ci] = findSurvivalDistribution(
            lifetime, dead, outs[:, ci]
        )

    loss = []
    for ci, cj in clusterPairsToSample:
        pairLoss = distroLoss(
            distros[ci], nSamples[ci], distros[cj], nSamples[cj]
        )
        loss.append(pairLoss)
    # for ci in range(len(uniqueClusters) - 1):
    #     sign = tf.math.sign(tf.reduce_mean(distros[ci]) - tf.reduce_mean(distros[ci + 1]))
    #     pairLoss = distroLoss(
    #         distros[ci], nSamples[ci], distros[ci + 1], nSamples[ci + 1]
    #     ) * sign
    #     loss.append(pairLoss)

    loss = tf.reduce_max(tf.stack(loss, 0))
    return loss

def distroLoss(distA, nA, distB, nB):
    assert distA.shape[0] == distB.shape[0], "Distributions of different length"
    effectiveN = tf.math.sqrt((nA * nB) / (nA + nB))

    Dplus = tf.clip_by_value(tf.math.reduce_max(distA - distB), 0, 1e5)
    Dminus = tf.clip_by_value(tf.math.reduce_max(distB - distA), 0, 1e5)
    V = Dplus + Dminus

    logloss = kuiperVariants(effectiveN, V)

    return logloss


def kuiperVariants(effectiveN, V):
    lam = (effectiveN + 0.155 + 0.24 / effectiveN) * V
    lambda_squared = lam ** 2

    sqrt2lam = 1 / (np.sqrt(2) * lam)
    r_lower = tf.math.floor(sqrt2lam)
    r_upper = tf.math.ceil(sqrt2lam)

    kuiperTerm1 = lambda r, lam_sq: 4 * r ** 2 * lam_sq - 1
    logKuiperTerm2 = lambda r, lam_sq: -2 * r ** 2 * lam_sq

    w = lambda r, lam_sq: -r * tf.math.exp(logKuiperTerm2(r, lam_sq))
    v = lambda r, lam_sq: kuiperTerm1(r, lam_sq) * tf.math.exp(
        logKuiperTerm2(r, lam_sq)
    )

    logloss = 0
    if r_lower >= 1:
        logloss = (
            logloss
            + w(r_lower, lambda_squared)
            - w(1, lambda_squared)
            + v(r_lower, lambda_squared)
        )

        logloss = logloss + v(r_upper, lambda_squared) - w(r_upper, lambda_squared)
        logloss = tf.math.log(logloss)
    else:
        logloss = (
            logloss
            + tf.math.log(kuiperTerm1(r_upper, lambda_squared))
            + logKuiperTerm2(r_upper, lambda_squared)
        )

    return logloss

def findSurvivalDistribution(lifetimes, deads, weights=None):
    if weights is None:
        weights = tf.ones_like(lifetimes, dtype=tf.float32)
    freq_lifetimes = tf.math.bincount(lifetimes, weights)
    freq_lifetimesDead = tf.math.bincount(lifetimes, weights * tf.cast(deads, tf.float32))
    nAlive = tf.reverse(tf.math.cumsum(tf.reverse(freq_lifetimes, [0])), [0])
    KMLambda = freq_lifetimesDead / nAlive
    KMProd = tf.math.cumprod(1 - KMLambda, 0)
    return KMProd